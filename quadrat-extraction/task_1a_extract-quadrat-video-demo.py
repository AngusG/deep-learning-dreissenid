import os
import cv2
import argparse
import numpy as np
import pandas as pd

from glob import glob

# local libraries
from utils import draw_lines


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hough lines video demo')
    parser.add_argument('--file', help='source video file name',
                        default='GLNI_12-1_2016-07-11_video-1.mp4')
    parser.add_argument('--datapath', help='root path to dataset',
                        default='/scratch/gallowaa/cciw/Data')
    parser.add_argument('--fps', help='frames per second for video output',
                        default=20)
    parser.add_argument('--votes', help='number of votes lines must have',
                        default=50, type=int)
    parser.add_argument('--mll', help='minimum line length in pixels',
                        default=400, type=int)
    parser.add_argument('--mlg', help='minimum line gap in pixels',
                        default=200, type=int)

    args = parser.parse_args()

    # Search for all video files on Google Drive...
    all_videos = glob(
        os.path.join(
            args.datapath, 'Videos_and_stills/GLNI/*/*/*/Videos/Quad*/*.mp4')
        )
    videotable_path = os.path.join(args.datapath, 'Tables', 'QuadratVideos.csv')
    video_df = pd.read_csv(videotable_path, index_col=0)

    vpath = video_df[video_df['Name'] == args.file]['Quadrat Video Path'].values[0]
    tokens = vpath.split('\\')  # windows backslash fmt for original HDD loc

    video_path = args.datapath + '/Videos_and_stills/GLNI'
    for tok in tokens[4:-1]:
        video_path += '/' + tok
    video_path = os.path.join(video_path, args.file)
    print(video_path)
    '''
    out_path = os.path.join(video_path, 'data')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
        print('Making dir ', out_path)
    '''
    cap = cv2.VideoCapture(video_path)
    #cap.set(cv2.CAP_PROP_FPS, FPS)

    raw_sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if cap.isOpened():
        print('Successfully opened video stream with resolution ', raw_sz)
        vout = cv2.VideoWriter()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        currentFrame = 0
        #for i in range(100):  #
        while(True):

            # Capture frame-by-frame
            ret, im = cap.read()
            if not ret: break

            # This format seems to have horizontal bars that trigger Hough lines
            if raw_sz[0] == 1440 and raw_sz[1] == 1080:
                y_trim = 145
                x_trim = 1
                im = im[y_trim:-y_trim, x_trim:-x_trim, :]

            # Do processing
            img, _ = draw_lines(
                im, rho=1, theta=np.pi/45, mll=args.mll, mlg=args.mlg,
                threshold=args.votes, ds=1)

            # Only on the first iterations
            if currentFrame == 0:
                sz = (img.shape[1], img.shape[0])
                vout.open(
                    args.file.split('.')[0] + '-quadrat-demo.mp4',
                    fourcc, args.fps, sz, True)

            """For annotating video

            @param org Bottom-left corner of the text string.
            @param org Bottom-left corner of the text string in the image.
            @param fontFace Font type, see #HersheyFonts.
            @param fontScale Font scale factor that is multiplied by the
                             font-specific base size.
            @param color Text color.
            @param thickness Thickness of the lines used to draw a text.
            @param lineType Line type. See #LineTypes
            """
            cv2.putText(
                img, str(currentFrame), (50, 50), # img, text, x, y coords,
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)  # font, 2, colour, line thickness,
            # write frame to video output
            vout.write(img)
            # To stop duplicate images
            currentFrame += 1
    else:
        print('Unable to open stream ', video_path)
# When everything done, release the capture
cap.release()
vout.release()
