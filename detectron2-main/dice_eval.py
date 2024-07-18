# load libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import skimage as ski
import os
from os import walk
import cv2 as cv
import glob
from miseval import evaluate
import json
import math
import shutil
import csv
import matplotlib as mpl


PRED_PATH = './datasets/coco/test_2017/predmasks/'
GT_PATH = './datasets/coco/test_2017_masks/'
TEST_IMG_PATH = './datasets/coco/test_2017/'
OUTPUT_PATH = "./datasets/coco/test_2017/overlays/"

EXT_FILTER = "/*.png"

PRED_COLOR = (255, 0, 0) # red
GT_COLOR = (0, 0, 255) # blue
INT_COLOR = (0, 255, 0) #green
ALPHA = 0.3

# clear output folder
try:
    shutil.rmtree(OUTPUT_PATH)
except:
    print('Output folder could not be cleared or did not exist')

# create output folders
os.makedirs(OUTPUT_PATH)

def dice_eval():

    dice_scores = []

    baseimages = glob.glob(TEST_IMG_PATH + EXT_FILTER)
    baseimages.sort()
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for fullpath in baseimages:

        imgname = os.path.basename(fullpath)

        try:
            predmask = read_mask(PRED_PATH + imgname)
            gtmask = read_mask(GT_PATH + imgname)        

        except:
            print('Prediction mask not found for: ' + imgname)
            continue

        dice_scores.append( (imgname, round(evaluate(gtmask, predmask, metric="DSC"),3) ) ) 

        outfilename = OUTPUT_PATH + imgname
        testimgdata = ski.io.imread(TEST_IMG_PATH + imgname)

        generate_overlay_img(testimgdata, predmask, gtmask, outfilename)

    dice_vals = save_scores(dice_scores)
    generate_histogram(dice_vals, bin_width=0.05)
    plot_loss()

def save_scores(dice_scores):
    dice_vals = np.array(dice_scores)[:,1].astype(float)
    dice_avg = round(np.mean(dice_vals),3)
    dice_median = round(np.median(dice_vals),3)
    dice_std = round(np.std(dice_vals),3)
    min_dice = dice_scores[np.argmin(dice_vals)]
    max_dice = dice_scores[np.argmax(dice_vals)]
                               
    with open('./save/dice_scores.csv', 'w') as f:    
        write = csv.writer(f) 
        write.writerow(['avg', 'median', 'std'])
        write.writerow([dice_avg, dice_median, dice_std])
        write.writerow(['Lowest DSC', 'Highest DSC'])
        write.writerow([min_dice, max_dice])
        write.writerows(dice_scores)

    return dice_vals

def plot_loss():
    total_loss = []
    tl_iter = []
    val_loss = []
    val_iter = []
    time = [0]
    with open('./output/metrics.json') as json_file:
        for line in json_file:
            cur_line = json.loads(line)
            if 'validation_loss' in cur_line:
                try:
                    val_iter.append(cur_line['iteration'])
                    val_loss.append(cur_line['validation_loss'])
                except:
                    continue

            try:
                tl_iter.append(cur_line['iteration'])
                total_loss.append(cur_line['total_loss'])
                time.append(float(cur_line['data_time'])+time[-1])
            except:
                continue

        tl_iter = np.array(tl_iter).astype(int)
        total_loss = np.array(total_loss).astype(float)
        val_iter = np.array(val_iter).astype(int)
        val_loss = np.array(val_loss).astype(float)
        tl_iter = tl_iter[:-1]
        time = time[1:]

        mpl.rcParams["font.size"] = 14
        fig, ax1 = plt.subplots(1,1)
        ax1.plot(tl_iter, total_loss, val_iter, val_loss)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Training Loss Curves')
        ax1.set_ylim(0,1)
        ax1.legend(['total loss', 'validation loss'])
        text_str = 'Min Val Loss: {:0.3f}'.format(np.min(val_loss)) + ' at {:d}'.format(val_iter[np.argmin(val_loss)])
        ax1.text(1, np.min(val_loss)*1.5, text_str)

        plt.savefig('./save/loss.png')
        plt.show()


def generate_histogram(scores, bin_width):
    bins = np.arange(0, 1.01, bin_width)

    mpl.rcParams["font.size"] = 14
    fig, ax = plt.subplots(1,1)
    ax.hist(scores, bins, edgecolor='black')
    ax.set_xlabel('Dice Similarity Coefficient (DSC)')
    ax.set_ylabel('Frequency')
    ax.set_xticks(bins)
    ax.set_xlim(math.floor((min(scores)*10))/10, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.axvline(scores.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(np.median(scores), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(scores.mean()*1, max_ylim*1.02, 'Mean: {:.3f}'.format(scores.mean()))
    plt.text(np.median(scores)*1, max_ylim*1.08, 'Median: {:.3f}'.format(np.median(scores)))
    plt.savefig('./save/hist.png')
    plt.show()


def generate_overlay_img(img, predmask, gtmask, outfilename):
    # ensure masks are grayscale
    if len(predmask.shape) > 3:
        predmask = cv.cvtColor(predmask, cv.COLOR_BGR2GRAY) 
    if len(gtmask.shape) > 3:       
       gtmask = cv.cvtColor(gtmask, cv.COLOR_BGR2GRAY) 

    intmask = cv.bitwise_and(predmask,gtmask)
    predmask = cv.bitwise_xor(predmask, intmask)
    gtmask = cv.bitwise_xor(gtmask, intmask)

    image_combined = overlay(img, intmask, predmask, gtmask, INT_COLOR, PRED_COLOR, GT_COLOR, ALPHA)

    cv.imwrite(outfilename, image_combined)


def read_mask(fullpath):
    mask = ski.io.imread(fullpath)
    mask[mask==255] = 1
    return mask


def overlay(image, intmask, predmask, gtmask, intcolor, predcolor, gtcolor, alpha, resize=None):
    # """Combines image and its segmentation mask into a single image.
    # https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    #     color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
    #     alpha: Segmentation mask's transparency. float = 0.5,
    #     resize: If provided, both image and its mask are resized before blending them together.
    #     tuple[int, int] = (1024, 1024))

    # Returns:
    #     image_combined: The combined image. np.ndarray

    # """
    intcolor = intcolor[::-1]
    intmask = np.expand_dims(intmask, 0).repeat(3, axis=0)
    intmask = np.moveaxis(intmask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=intmask, fill_value=intcolor)
    int_overlay = masked.filled()

    predcolor = predcolor[::-1]
    predmask = np.expand_dims(predmask, 0).repeat(3, axis=0)
    predmask = np.moveaxis(predmask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=predmask, fill_value=predcolor)
    pred_overlay = masked.filled()

    gtcolor = gtcolor[::-1]
    gtmask = np.expand_dims(gtmask, 0).repeat(3, axis=0)
    gtmask = np.moveaxis(gtmask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=gtmask, fill_value=gtcolor)
    gt_overlay = masked.filled()

    if resize is not None:
        image = cv.resize(image.transpose(1, 2, 0), resize)
        pred_overlay = cv.resize(pred_overlay.transpose(1, 2, 0), resize)
        gt_overlay = cv.resize(gt_overlay.transpose(1, 2, 0), resize)

    image_combined = cv.addWeighted(image, 1 - alpha, int_overlay, alpha, 0)
    image_combined = cv.addWeighted(image_combined, 1 - alpha, pred_overlay, alpha, 0)
    image_combined = cv.addWeighted(image_combined, 1 - alpha, gt_overlay, alpha, 0)

    return image_combined

dice_eval()
