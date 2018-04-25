from music21 import *
import numpy as np
import Preprocess
from RecurrentNN import *
import ReadMidi
from pyspark import SparkContext, SparkConf
import argparse



def Begin(file):
    # Read in Midi organize data

    Instruments = Preprocess.getInstruments(file)
    dictOfNotes = Preprocess.getNotes(Instruments)
    for index in range(len(Instruments)):
        iterations = 100
        learningRate = 0.001
        # load input output data
        returnData, numCategories, expectedOutput, outputSize, data = ReadMidi.LoadText(dictOfNotes, index)
        RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)

        # training time!
        for i in range(1, iterations):
            # compute predicted next word
            RNN.forwardProp()
            # update all our weights using our error
            error = RNN.backProp()
            # once our error/loss is small enough
            print("Error on iteration ", i, ": ", error)
            if error > -100 and error < 100 or i % 100 == 0:
                # we can finally define a seed word
                seed = np.zeros_like(RNN.x)
                maxI = np.argmax(np.random.random(RNN.x.shape))
                seed[maxI] = 1
                RNN.x = seed
                # and predict some new text!
        output = RNN.sample()
        # write it all to disk
        ReadMidi.ExportMidi(output, data, Instruments, index, file)


if __name__ == "__main__":
    conf = (SparkConf()
            .setMaster("local")
            .setAppName("My app")
            .set("spark.executor.memory", "1g"))
    sc = SparkContext(conf=conf)

    parser = argparse.ArgumentParser("\nParse input Arguments for LSTM Music Prediction")
    parser.add_argument("-f", "--file", help="Add file path for Midi files\n", type=str)
    args = vars(parser.parse_args())
    file = args["file"]
    f = open(file, "r")
    all_files = [line for line in f]
    f.close()

    all_files = sc.parallelize(all_files, 10)
    all_files.foreach(Begin)


