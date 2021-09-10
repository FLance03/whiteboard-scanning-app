import heapq

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from Steps import (SaturationAndOutlier2 as Step2,
                   ConnectedComponentGrouping5 as Step5,
                   FeatureExtraction6 as Step6)
from testing import testing

MINHEIGHTPROP = 0.8

def FitWithinThresh(first, second):
    top1, bottom1 = first[2], first[3]
    top2, bottom2 = second[2], second[3]
    lesserTop, greaterTop = (top1, top2) if top1 < top2 else (top2, top1)
    lesserBottom, greaterBottom = (bottom1, bottom2) if bottom1 < bottom2 else (bottom2, bottom1)
    return lesserBottom - greaterTop >= MINHEIGHTPROP * (greaterBottom - lesserTop)


def main(imgsFeatures, imgsPhrases, imgsNonTextLabels):
    # imgsFeatures is a 3d array (not including the numpy arrays inside)
    redundantCC = []
    pastImgFeatures = []
    assert len(imgFeatures) > 0, "len(imgFeatures) == 0"
    # Copy the first index of imgFeatures to initialize and successively compare with the next indices
    # np array is not deep copied
    pastImgFeatures.append(imgFeatures[0].copy())
    # For each image...
    for imageNum, currentImgFeatures in enumerate(imgsFeatures[1:]):
        for ind, feature in enumerate(currentImgFeatures):
            # Keep track of the indices when currentImgFeatures was sorted by top
            feature['index'] = ind
        # Sort by minimum height requirements
        currentImgFeatures = sorted(currentImgFeatures, key=lambda row:row['minHeight'])
        assert imageNum == len(pastImgFeatures)
        # pastImageInds holds the current index of each nonRedundantCC which was sorted by top
        #    which supplies the need of the each currentImgFeatures at the 'minHeight' key which was also sorted
        currentImgInd, pastImageInds = 0, [0] * len(pastImgFeatures)
        # redundantHeap stores all CC in the current image that is considered redundant with
        redundantHeap = []
        # For each connected component in that image...
        while currentImgInd < len(currentImgFeatures):
            assert left<=right and top<=bottom
            currentCCFeatures = currentImgFeatures[currentImgInd]
            minHeightNeed = currentCCFeatures['minHeight']
            maxHeightNeed = bottom + minHeightNeed
            # For each previous image...
            for pastImgNum, pastImgInd in enumerate(pastImageInds):
                # For each connected component of that previous image...
                while pastImgInd < len(pastImgFeatures[pastImgNum]) and pastImgInd <= maxHeightNeed:
                    if FitWithinThresh(nonRedundantCC[pastImgNum][pastImgInd]['info'],
                                            currentCCFeatures['info']):
                        isRedundant, degreeOfRedundancy, pastFeatureMatches, currentFeatureMatches = \
                            GetDegreeOfRedundancy(nonRedundantCC[pastImgNum][pastImgInd], currentCCFeatures,
                                                  imgsPhraseLabels if currentCCFeatures['type'] == 'phrase'
                                                                    else imgsNonTextLabels)
                        if isRedundant:
                            # Maxheap based on degreeOfRedundancy
                            heapq.heappush(redundantHeap, (-degreeOfRedundancy, pastRedundancyLabel, currentRedundancyLabel,
                                                            currentCCFeatures['index'], pastImgNum, pastImgInd))
                    # We increment pastImgInd but when the nesting for loop of this while loop runs again, there
                    #    is no change since the succeeding currentImgInd may still be True based on FitWithinThresh
                    #    for this pastImageInds[pastImgInd]
                    pastImgInd += 1
                if pastImageInds[pastImgNum] == pastImgInd:
                    # The while loop above did not even run at least once so we increment pastImageInds[pastImgNum]
                    #    since pastImageInds is sorted by the key top which is what is being sought after by
                    #    each element of currentImgFeatures. currentImgFeatures is sorted by the key minHeight
                    #    and is thus sorted by the needs of its elements (supplied by the pastImageInds)
                    pastImageInds[pastImgNum] += 1
        redundants = UpdateFeatureInfo(imgsFeatures, redundantHeap, imgsPhraseLabels, imgsNonTextLabels)
        redundantCC.append(redundants)
        # Cannot append currentImgFeatures instead of imgsFeatures[imageNum] since currentImgFeatures is
        #    already sorted by the minHeight key instead of the top key
        pastImgFeatures.append(imgsFeatures[imageNum])
        for CCInfo in currentImgFeatures:
            left, right, top, bottom = CCInfo[1][:4]

        for pastImages in nonRedundanctCC:
            # pastFeatures hold all features that are processed before the current one (currentFeatures)
            #    which are still not recognized as redundant


testImages = ['redundancyfrontcropped', 'redundancyleftcropped', 'redundancyrightcropped']
imgsFeatures = []
imgsPhraseLabels = []
imgsNonTextLabels = []
for img in testImages:
    file = np.load(img + '.npz')
    labels, labelsInfo, textNonText, textLabels, wordLabels, phraseLabels, nonTextLabels = \
        file['labels'], file['labelsInfo'], file['textNonText'], file['textLabels'], file['wordLabels'], file[
            'phraseLabels'], file['nonTextLabels']
    imgsPhraseLabels.append(phraseLabels)
    imgsNonTextLabels.append(nonTextLabels)
    imgFeatures = []
    index = 0
    for ctr in range(2):
        currentLabels = phraseLabels if ctr==0 else nonTextLabels
        labelsInfo = Step5.GetLabelsInfo(currentLabels)
        for ind, phraseLabelInfo in enumerate(labelsInfo):
            imgFeatures.append({})
            imgFeatures[-1]['labelNum'] = ind + 1
            imgFeatures[-1]['type'] = 'phrase' if ctr==0 else 'nonText'
            left, right, top, bottom = phraseLabelInfo[:4]
            # Include top as the first element of each row.
            # top will be used to identify the possible CCs to compare
            # imgFeatures will also be sorted by top later
            imgFeatures[-1]['top'] = top
            width, area = phraseLabelInfo[7], phraseLabelInfo[9]
            imgFeatures[-1]['minHeight'] = top - ((1 - MINHEIGHTPROP) * area // width)

            # Assert that currentLabels does not have any other position with the label number ind+1 other
            #    than what is inside the given rectangle
            checkLabels = currentLabels.copy()
            checkLabels[top:bottom+1, left:right+1] = 0
            assert len(checkLabels[checkLabels == ind+1]) == 0

            # Indexing currentLabels is not exactly required by Step6.main() but since inside it, the currentLabels is
            #    again filtered out to only include the ind+1 label and further getting the info of such label (for a general
            #    case instead of passing labelInfo as another parameter), reducing the search space may improve performance
            # Array is also not modified in Step6.main() since it uses np.where at the onset
            # ind + 1 since Step5.GetLabelsInfo() does not include label 0 thus, label 1 has index zero as info
            features, info = Step6.main(currentLabels[top:bottom+1, left:right+1], ind+1)
            imgFeatures[-1]['info'] = info
            imgFeatures[-1]['features'] = features
            imgFeatures[-1]['index'] = index
            index += 1
    # Base the sorting on the first element (data about top position) of each row
    imgsFeatures.append(sorted(imgFeatures, key=lambda row:row['top']))
main(imgsFeatures, imgsPhraseLabels, imgsNonTextLabels)
# phraseLabelsInfo = Step5.GetLabelsInfo(phraseLabels)
# features = Step6.main(phraseLabels, phraseLabelsInfo[0], 1)
# testing.FullPrint(features)

# cv.imshow('Labels', testing.imshow_components(labels))
# cv.imshow('Labels2', testing.imshow_components(np.where(phraseLabels == 1, phraseLabels, 0)))
# cv.imshow('Texts', testing.imshow_components(textLabels))
# cv.imshow('Words', testing.imshow_components(wordLabels))
# cv.imshow('Phrases', testing.imshow_components(phraseLabels))
# cv.imshow('Non Texts', testing.imshow_components(nonTextLabels))
#
cv.waitKey()
cv.destroyAllWindows()