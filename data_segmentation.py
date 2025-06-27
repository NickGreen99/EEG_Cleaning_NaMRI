import mne
import numpy as np
from subject_segments import subject_segments
import glob
import re
import gc
from pathlib import Path

def epoch_data(eeg_segment, window_size_time):
  s_freq = eeg_segment.info["sfreq"]
  window_size = int(window_size_time * s_freq)

  # Sample indices from 0 up to len(eeg_segment)
  steady_samples = np.arange(0, len(eeg_segment) - window_size, window_size)
  events = np.vstack([ steady_samples + eeg_segment.first_samp,
                      np.zeros(len(steady_samples)),
                      np.ones(len(steady_samples))]).astype(int).T
  
  # Create an epochs object with our events
  epochs = mne.Epochs(eeg_segment, events, tmin=0, tmax=window_size_time-(1/s_freq),
                      baseline=(0, 0), reject=None, flat=None,
                      reject_by_annotation=False, verbose=False)
  return epochs.get_data()

def eeg_read(vhdr_file):
    
    window_size_time = 0.125

    chs = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','FPz','Fz','Cz','Pz',
    'POz','Oz','FT9','FT10',"TP9'","TP10'"
    ]
    
    # read raw recording
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=False)

    match = re.search(r'(\d{3})_', vhdr_file)
    if match:
        subject_id = int(match.group(1))
    
    SUBJECT_DIR = OUTPUT_DIR / str(subject_id)
    SUBJECT_DIR.mkdir(parents=True, exist_ok=True)

    raw.pick(chs)
    raw.resample(512.)
    raw.set_annotations(None)
    
    clean_segments = []
    dirty_segments = []
    noise_segments = []

    # Clean Array Writing
    print('#### Writing Clean Segments ... ####')
    for clean_time_slot in subject_segments[subject_id]['clean']:
        seg = raw.copy().crop(tmin=clean_time_slot[0], tmax=clean_time_slot[1])
        data = epoch_data(seg, window_size_time)
        clean_segments.append(data)
        del seg, data
        gc.collect()

    # Concatenate all epochs along axis 0 (n_epochs)
    clean_segments = np.concatenate(clean_segments, axis=0)
    
    # Save subject's clean.npy
    out_path = SUBJECT_DIR / 'clean.npy'
    np.save(out_path, clean_segments)
    
    del clean_segments
    gc.collect()

    # Dirty Array Writing
    print('#### Writing Dirty Segments ... ####')
    for dirty_time_slot in subject_segments[subject_id]['dirty']:
        dirty_seg = raw.copy().crop(tmin=dirty_time_slot[0], tmax=dirty_time_slot[1])
        dirty_data = epoch_data(dirty_seg, window_size_time)
        dirty_segments.append(dirty_data)

        # Keep only the noise
        noise_proxy_seg = dirty_seg.filter(l_freq=100., h_freq=250., fir_design='firwin', verbose=False)
        noise_data = epoch_data(noise_proxy_seg, window_size_time)
        noise_segments.append(noise_data)

        del dirty_seg, noise_proxy_seg, dirty_data, noise_data
        gc.collect()

    # Concatenate all epochs along axis 0 (n_epochs)
    dirty_segments = np.concatenate(dirty_segments, axis=0)
    noise_segments = np.concatenate(noise_segments, axis=0)

    # Save subject's dirty.npy
    out_path = SUBJECT_DIR / 'dirty.npy'
    np.save(out_path, dirty_segments)

    # Save subject's noise.npy
    out_path = SUBJECT_DIR / 'noise.npy'
    np.save(out_path, noise_segments)
       
    del dirty_segments, noise_segments
    gc.collect()

def main():
    data_paths = glob.glob("data_raw/*.vhdr")
    for subject_vhdr in data_paths:
        eeg_read(subject_vhdr)

if __name__ == '__main__':
    OUTPUT_DIR = Path("data_segmented")
    main()
