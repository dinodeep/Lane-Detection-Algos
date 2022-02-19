import argparse
import cv2 as cv

import detectionAlgos as detect


def main(args):

    # get files
    infile = args.infile
    outfile = args.outfile
    filename, filetype = outfile.split(".")

    # use various methods (currently only 1) to detect the lines
    img = cv.imread(infile)
    linedImg0 = detect.method0(img)

    # save images
    cv.imwrite(filename + "." + filetype, linedImg0)


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="provide parameters to run lane detection algorithms on image")
    parser.add_argument("infile")
    parser.add_argument("outfile")

    args = parser.parse_args()
    main(args)
