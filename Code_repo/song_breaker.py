#!/usr/bin/env python
# coding: utf-8

# In[144]:


def song_break(y,sr):
    
    import librosa
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Audio
    from librosa import display
    import numpy as np
    import os

    from pathlib import Path
    from pydub import AudioSegment

    print("starting conversion...")

    data_folder = Path("E:/DS/Projects/tunes_repo/MIRaGE/Song_repo/")

    music_list = os.listdir(data_folder / 'Mirage_wav')

    music_wav = "E:/DS/Projects/tunes_repo/MIRaGE/Song_repo/Mirage_wav"

    curr_track = music_wav + "/" + str(music_list[0])

    y, sr = librosa.load(curr_track)


    print("checking autocorrelation...")

    s_corr = []

    for i in range(0,len(y), sr*20):
        x = y[i:i+(sr*20)]
        i_corr = []
        for j in range(0,len(y), sr*20):

            if i == j:
                pass
            else:
                yhat = y[j:j+(sr*20)]

                corr = np.correlate(np.abs(x), np.abs(yhat))
                i_corr.append(corr.mean())

        s_corr.append(i_corr)


    df = pd.DataFrame(s_corr)

    print("breaking snippets...")

    plot_df = pd.DataFrame(df.mean())
    plot_df.columns = ['corr_mean']
    plot_df['section'] = "Intro"
    plot_df['min_time'] = 0
    plot_df['max_time'] = 0
    section_count = 0
    chorus_count = 0

    section = 0

    i_max = 0


    plot_df.section = np.where(plot_df.corr_mean > np.percentile(plot_df.corr_mean,80), "Chorus", plot_df.section)

    plot_df['min_time'] = plot_df.index*sr*20
    plot_df['max_time'] = (plot_df.index+1)*sr*20

    plot_df['prev_corr'] = plot_df.corr_mean.shift(1)
    plot_df['delta'] = np.abs((plot_df.prev_corr - plot_df.corr_mean)/plot_df.prev_corr)


    chorus_count = 1
    section_count = 1
    plc = 0
    plot_df['section_copy'] = 0

    for snippet in range(1,len(plot_df)):

        if plot_df.section[snippet] == "Chorus":
            plc = chorus_count
            section_name  = "Chorus" + str(plc)
            if plot_df.section[snippet-1] == "Chorus":
                plot_df.section_copy[snippet] = plot_df.section_copy[snippet-1]
            else: 
                plot_df.section_copy[snippet] = section_name
                chorus_count+=1

        else:

            plc = section_count
            section_name  = "Section" + str(plc)

            if plot_df.delta[snippet] < 0.15:
                plot_df.section_copy[snippet] = plot_df.section_copy[snippet-1]
            else: 
                plot_df.section_copy[snippet] = section_name
                section_count+=1

        if snippet == len(plot_df)-1:
            plot_df.section_copy[snippet] = "outro"
        else:
            pass

    plot_df.section = np.where(plot_df.section_copy == 0,plot_df.section, plot_df.section_copy)

    section_df = plot_df[['section','min_time','max_time']].groupby(['section']).agg({'min_time':'min','max_time':'max'}).reset_index()

    print("song broken into " + str(len(section_df)) + " snippets...")

    return section_df

