import numpy as np
from music21 import *

# Get all Bach chorale files
bach_chorales = corpus.getComposer('bach')

# File to store the results
output_file = 'data/preprocess.txt'
subdivision = 0.5

with open(output_file, 'w') as f:
    for chorale in bach_chorales:
        try:
            s = corpus.parse(chorale)
            k = s.analyze('key')
            
            if k.mode == 'minor':
                target_key =  pitch.Pitch('A')  # Transpose to A if the key is minor
            else:
                target_key =  pitch.Pitch('C')  # Transpose to C if the key is major
            i = interval.Interval(k.tonic, target_key)
            s = s.transpose(i)
            # print(f'Key: {s.analyze('key')}')

            parts = s.parts
            if len(parts) > 4: 
                continue
            measure = parts[0].getElementsByClass('Measure')[0]  

            time_signature = measure.timeSignature
            measure_length = measure.quarterLength
            # print(f'Time signature: {time_signature} | measure_length: {measure_length}')

            if measure_length == 1:
                measure_length = 4 

            num_slots = int(measure_length / subdivision)
            data = []

            for measure_index in range(len(parts[0].getElementsByClass('Measure'))):
                shape = (1, num_slots, len(parts))
                out = np.empty(shape, dtype=object)
                
                for part_index, part in enumerate(parts):
                    measure = part.measure(measure_index + 1)
                    if measure is not None:
                        offset = 0.0
                        count = 0
                        while offset < measure_length:
                            # Iterate over both notes and rests in the measure
                            element = next((el for el in measure.notesAndRests if el.offset == offset), None)
                            
                            if element:
                                if isinstance(element, note.Note):  # Handle note
                                    token = f"p{part_index}{element.nameWithOctave}"
                                elif isinstance(element, note.Rest):  # Handle rest
                                    token = f"p{part_index}|"
                                
                                # Check for fermata
                                for exp in element.expressions:
                                    if isinstance(exp, expressions.Fermata):
                                        token += '·'
                                
                                out[0][count][part_index] = token
                            else:
                                out[0][count][part_index] = f"p{part_index}"
                            
                            # Move the offset forward by 0.25 (quarter note duration)
                            offset += subdivision
                            count += 1
                if np.all(out.flatten() == None):
                    continue
                else:
                    data.extend(out.flatten())  # Add to data if not all are None
            # Write to file
            f.write(" ".join(data) + "\n" )


        except Exception as e:
            print(f"Error processing {chorale}: {e}")
            continue  # Skip to the next chorale if an error occurs

        print(f"Processing {chorale} ")

print(f"Processing complete. Data saved to {output_file}")

