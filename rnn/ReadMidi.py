

def LoadText(dictInstrumentNotes, index):
    import numpy as np
    # open text and return input and output data (series of words)
    data = dictInstrumentNotes[index]
    text = data
    outputSize = len(text)
    data = list(set(text))
    uniqueWords, dataSize = len(data), len(data)
    returnData = np.zeros((uniqueWords, dataSize))
    for i in range(0, dataSize):
        returnData[i][i] = 1
    returnData = np.append(returnData, np.atleast_2d(data), axis=0)
    output = np.zeros((uniqueWords, outputSize))
    for i in range(0, outputSize):
        index = np.where(np.asarray(data) == text[i])
        output[:, i] = returnData[0:-1, index[0]].astype(float).ravel()
    return returnData, uniqueWords, output, outputSize, data


# write the predicted output (series of words) to disk
def ExportMidi(output, data, Instruments, index, file):
    import sys
    sys.path.insert(0, "/s/bach/c/under/btroksa/.local/lib/python3.6/site-packages/")
    import music21
    import numpy as np
    prob = np.zeros_like(output[0])
    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            prob[j] = output[i][j] / np.sum(output[i])
        outputNote = np.random.choice(data, p=prob)
        if (len(outputNote) > 4):
            Instruments[index].getElementsByClass(music21.stream.Part)[0].append(music21.chord.Chord(outputNote))
        else:
            Instruments[index].getElementsByClass(music21.stream.Part)[0].append(music21.note.Note(outputNote))
    WriteInstruments(Instruments, file)


def WriteInstruments(Instrum, file):
    import sys
    sys.path.insert(0, "/s/bach/c/under/btroksa/.local/lib/python3.6/site-packages/")
    import music21
    outputMidi = music21.stream.Score()
    for i in Instrum:
        tempStream = music21.stream.Part()
        for j in i.getElementsByClass(music21.stream.Part):
            tempStream.append(j)
        outputMidi.append(tempStream)
        try:
            mf = music21.midi.translate.streamToMidiFile(outputMidi)
            mf.open("file://"+file[:-4] + "_new.mid", 'wb')
            mf.write()
            mf.close()
        except:
            pass
    return

