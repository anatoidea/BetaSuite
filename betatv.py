from pathlib import Path
import onnxruntime
import cv2
import betaconst
import betautils
import betaconfig
import numpy as np
import os
import json
import math
import time
import hashlib
import subprocess as sp

cap = cv2.VideoCapture()

censor_hash = betautils.get_censor_hash()
session = betautils.get_session()

parts_to_blur = betautils.get_parts_to_blur()

images_to_censor = []
images_to_detect = []

has_audio = False

to_censor = []
to_detect = []

session = betautils.get_session()

for root,d_names,f_names in os.walk(betaconst.video_path_uncensored):
    censored_folder = root.replace( betaconst.video_path_uncensored, betaconst.video_path_censored, 1 )
    os.makedirs( censored_folder, exist_ok=True )

    print( "Processing %s"%(root) )

    for fidx, fname in enumerate(f_names):
        #try:
        if 1==1:
            (stem, suffix ) = os.path.splitext(fname)

            uncensored_path = os.path.join( root, fname )

            cap.open( uncensored_path )
            if cap.isOpened():
                file_hash = betautils.md5_for_file( uncensored_path, 16 );

                censored_path = os.path.join( censored_folder, '%s-%s-%s-%s-%d-%.3f%s'%(stem, file_hash, censor_hash, "+".join(map(str,betaconfig.picture_sizes)), betaconfig.video_censor_fps, betaconst.global_min_prob, ".mp4"))
                censored_avi  = os.path.join( censored_folder, '%s-%s-%s-%s-%d-%.3f%s'%(stem, file_hash, censor_hash, "+".join(map(str,betaconfig.picture_sizes)), betaconfig.video_censor_fps, betaconst.global_min_prob, ".avi"))

                t1 = time.perf_counter() 
                if( not os.path.exists( censored_path ) ):
                    print( "Processing %d/%d: %s%s...."%(fidx+1, len( f_names ), stem, suffix) )
                    vid_fps = cap.get( cv2.CAP_PROP_FPS )
                    ret, frame = cap.read()
                    vid_h,vid_w,ch = frame.shape
                    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT )

                    all_raw_boxes = []
                    for size in betaconfig.picture_sizes:
                        box_hash_path = '../vid_hashes/%s-%s-%d-%d-%.3f.txt'%(file_hash,betaconst.picture_saved_box_version, size, betaconfig.video_censor_fps, betaconst.global_min_prob)

                        if( os.path.exists( box_hash_path ) ):
                            all_raw_boxes.append( json.load(open(box_hash_path)) )
                        else:
                            raw_boxes = []
                            cap.set( cv2.CAP_PROP_POS_FRAMES, 0 )
                            i = 0
                            t_pos = 0
                            while( cap.isOpened ):
                                ret, frame = cap.read()
                                if not ret:
                                    break

                                print( "size %d: processing frame %d/%d"%(size, cap.get(cv2.CAP_PROP_POS_FRAMES ), num_frames ), end="\r" )
                                this_raw_boxes = betautils.raw_boxes_for_img( frame, size, session, t_pos )

                                raw_boxes.extend( this_raw_boxes )

                                i += 1
                                t_pos = i / betaconfig.video_censor_fps
                                cap.set( cv2.CAP_PROP_POS_FRAMES, math.floor( t_pos * vid_fps ) )

                            json.dump( raw_boxes, open( box_hash_path, 'w' ) )
                            all_raw_boxes.append( raw_boxes )
                            print( "size %d: processing complete....................."%size )

                    boxes = [];
                    for raw_boxes in all_raw_boxes:
                        for raw in raw_boxes:
                            res = betautils.process_raw_box( raw, vid_w, vid_h )
                            if res:
                                boxes.append( res )

                    boxes.sort( key = lambda x: x['start'] )

                    command = [ "../ffmpeg/bin/ffmpeg.exe",
                            '-y',
                            '-hide_banner',
                            '-loglevel', 'error',
                            '-f', 'rawvideo',
                            '-vcodec','rawvideo',
                            '-s', '{}x{}'.format(vid_w,vid_h),
                            '-pix_fmt', 'bgr24',
                            '-r', '%.6f'%vid_fps,
                            '-i', '-',
                    ]

                    has_audio = betautils.video_file_has_audio( uncensored_path )

                    if betautils.video_file_has_audio( uncensored_path ):
                        command.extend( [
                            '-i', uncensored_path,
                            '-c:a', 'copy',
                            '-c:v', 'mpeg4',
                            '-qscale:v', '3',
                            '-map', '0:0',
                            '-map', '1:1',
                            '-shortest',
                            censored_avi
                        ] )
                    else:
                        command.extend( [
                            '-an',
                            '-c:v', 'mpeg4',
                            '-qscale:v', '3',
                            censored_avi
                        ] )

                    proc = sp.Popen(command, stdin=sp.PIPE )

                    cap.set( cv2.CAP_PROP_POS_FRAMES, 0 )
                    i=0
                    live_boxes = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        curr_time = i/vid_fps
                        for j,box in enumerate(live_boxes):
                            if box['end'] < curr_time:
                                live_boxes.pop(j)

                        while len( boxes ) and boxes[0]['start'] <= curr_time:
                            live_boxes.append( boxes.pop(0) )

                        frame = betautils.censor_img_for_boxes( frame, live_boxes )

                        proc.stdin.write(frame.tobytes())
                        i+=1
                        print( "encoded %d/%d frames"%(i,num_frames), end="\r" )

                    proc.stdin.close()
                    proc.wait()

                    print( "encoding complete, re-encoding to final output.........." );
                    command =  [ '../ffmpeg/bin/ffmpeg.exe',
                            '-y',
                            '-hide_banner',
                            '-loglevel', 'error',
                            '-stats',
                            '-i', censored_avi,
                            '-c', 'copy',
                            '-c:v', 'libx264',
                            '-crf', '23',
                            '-preset', 'veryfast',
                            censored_path
                    ]

                    proc2 = sp.Popen( command ) 
                    proc2.wait()
                    os.remove( censored_avi )

                else:
                    print( "--- Skipping  %d/%d (exists): %s"%(fidx+1, len(f_names), fname ) )

            else:
                print( "--- Skipping  %d/%d (not video): %s"%(fidx+1, len(f_names), fname ) )
        #except BaseException as err:
        if 1==0:
            print( "--- Skipping  %d/%d (failed)  [----- ----- ----- -----: -----]: %s"%(fidx+1, len(f_names), fname ) )
            print( f"Error {err=}, {type(err)=}" )
            time.sleep( 1 )