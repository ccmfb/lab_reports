import cv2
import numpy as np
import pandas as pd
import sys

def main():
    np.set_printoptions(threshold=sys.maxsize)
    paths = np.loadtxt('paths.txt', dtype=str)

    positions = []
    frames_per_second = []
    for count, path in enumerate(paths):
        pos, fps = get_positions(path, count+1, OUTPUT=1)
        positions.append(pos)
        frames_per_second.append(fps)

    positions = np.array(positions, dtype=object)
    frames_per_second = np.array(frames_per_second)

    dpos = {'pos': positions}
    dfps = {'fps': frames_per_second}
    dfpos = pd.DataFrame(data=dpos)
    dffps = pd.DataFrame(data=dfps)
    dfpos.to_csv('position_data.csv', sep=',', encoding='utf-8')
    dffps.to_csv('fps_data.csv', sep=',', encoding='utf-8')
    

def get_positions(path: str, count, OUTPUT = 1):
    cap = cv2.VideoCapture(path)
    object_detector = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=900)

    start = 0
    end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    processed_count = 0  # a count of how many frames we have saved 
    fps = cap.get(cv2.CAP_PROP_FPS)
    every = 1 / fps 

    pos = []
    while frame < end:  # lets loop through the frames until the end

        _, frame_orig = cap.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if frame_orig is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if round(frame % every) == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            pos.append(compute_pos(frame_orig, object_detector, count, frame))
            processed_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    pos = np.array(pos, dtype=object)
    print(f'Total frames = {end}, Processed Frames = {processed_count}, Frames Lost = {end - processed_count}')

    cap.release()
    cv2.destroyAllWindows()

    return pos, fps 


def compute_pos(frame_orig, object_detector, count, k, OUTPUT=1):
    # Defining a region of interest
    roi = frame_orig[1:500, 1:2 * 1280]

    # Changing the colours to make contour detection easier
    frame_grey = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Applying the background subtractor algorithm
    mask = object_detector.apply(frame_grey)

    # Finding contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Looping through all the found contours
    pos = []
    for cnt in contours:
        area = cv2.contourArea(cnt) # Getting area of contour

        if area > 500:
            centre, radius = cv2.minEnclosingCircle(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            #print(f'Current Position (int) = {int(centre[0]), int(centre[1])}')
            pos.append([k, centre[0], centre[1], w/2, h/2]) # Adding to pos list

            # Drawing contours and enclosing circle as well as rectangle for uncertainties
            cv2.drawContours(frame_orig, [cnt], -1, (0,0,0), 1)
            cv2.circle(frame_orig, (int(centre[0]), int(centre[1])), int(radius), (0,255,0), 2)
            cv2.rectangle(frame_orig, (x,y), (x+w,y+h), (0,255,0), 2)

    # Show videos
    if OUTPUT == 1:
        cv2.imwrite(f'process/vid_{count}/mask_{k}.jpg', mask)
        cv2.imwrite(f'process/vid_{count}/grey_{k}.jpg', frame_grey)
        cv2.imwrite(f'process/vid_{count}/orig_{k}.jpg', frame_orig)

    pos = np.array(pos)
    return pos[0] if pos.size != 0 else [k, -1, -1, -1, -1]


if __name__ == '__main__':
    main()

