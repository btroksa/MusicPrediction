



def getNotes(Instruments):
    import sys
    sys.path.insert(0, "/s/bach/c/under/btroksa/.local/lib/python3.6/site-packages/")
    import music21
    dictInstrumentNotes = {}
    for index, i in enumerate(Instruments):
        all_notes = []
        for k in i.getElementsByClass(music21.stream.Part):
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
    import sys
    sys.path.insert(0, "/s/bach/c/under/btroksa/.local/lib/python3.6/site-packages/")
    import music21
    midi_data = music21.converter.parse(file)
    Instruments = []
    for k in midi_data:
        try:
            Instruments.append(music21.instrument.partitionByInstrument(k))
        except:
            pass
    return Instruments
