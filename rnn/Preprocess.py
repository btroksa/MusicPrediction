from music21 import *

def getNotes(Instruments):
    dictInstrumentNotes = {}
    for index, i in enumerate(Instruments):
        all_notes = []
        for k in i.getElementsByClass(stream.Part):
            for nn in k.notes:
                if(nn.isNote):
                    all_notes.append(nn.nameWithOctave)
                elif(nn.isChord):
                    notes_in_chord = ""
                    for pitch in nn.pitches:
                        notes_in_chord += pitch.nameWithOctave + " "
                    all_notes.append(notes_in_chord)
        dictInstrumentNotes[index] = all_notes
    return dictInstrumentNotes

def getInstruments(file):
    midi_data = converter.parse(file)
    Instruments = []
    for k in midi_data:
        try:
            Instruments.append(instrument.partitionByInstrument(k))
        except:
            pass
    return Instruments
