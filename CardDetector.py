############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a PiCamera video feed.
#

# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards
import VideoStream
import platform


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:
# Automatically use USB camera on Mac
camera_mode = 2 if platform.system() == 'Darwin' else 1
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,camera_mode,0).start()
time.sleep(1) # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0 # Loop control variable

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
	
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card, cnt_is_stacked = Cards.find_stacked_cards(pre_proc)

    # If there are no contours, do nothing
    if len(cnts_sort) != 0:
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                
                # Check if this is a stacked card
                if cnt_is_stacked[i] == 1:
                    # Process as stacked cards
                    separated_cards = Cards.enhanced_preprocess_card(cnts_sort[i], image, is_stacked=True)
                    
                    for card in separated_cards:
                        # Find the best rank and suit match for each card
                        card.best_rank_match, card.best_suit_match, card.rank_diff, card.suit_diff = Cards.match_card(card, train_ranks, train_suits)
                        cards.append(card)
                        k += 1
                else:
                    # Process as single card (original logic)
                    card = Cards.preprocess_card(cnts_sort[i], image)
                    card.best_rank_match, card.best_suit_match, card.rank_diff, card.suit_diff = Cards.match_card(card, train_ranks, train_suits)
                    cards.append(card)
                    k += 1

        # Draw results for all detected cards
        for card in cards:
            image = Cards.draw_results(image, card)
        
        # Draw contours
        if len(cards) != 0:
            temp_cnts = []
            for card in cards:
                temp_cnts.append(card.contour)
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)
        
        
    # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
    # so the first time this runs, framerate will be shown as 0.
    cv2.putText(image,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

    # Finally, display the image with the identified cards!
    cv2.imshow("Card Detector",image)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
        

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
videostream.stop()

