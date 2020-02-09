{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def song_break(p):\n",
    "    import librosa\n",
    "    import pandas as pd\n",
    "    import matplotlib.pyplot as plt\n",
    "    from IPython.display import Audio\n",
    "    from librosa import display\n",
    "    import numpy as np\n",
    "    import os\n",
    "\n",
    "    from pathlib import Path\n",
    "    from pydub import AudioSegment\n",
    "    \n",
    "    data_folder = Path(\"E:/DS/Projects/tunes_repo/MIRaGE/Song_repo/\")\n",
    "\n",
    "    music_list = os.listdir(data_folder / 'Mirage_wav')\n",
    "\n",
    "    music_wav = \"E:/DS/Projects/tunes_repo/MIRaGE/Song_repo/Mirage_wav\"\n",
    "\n",
    "    curr_track = music_wav + \"/\" + str(music_list[p])\n",
    "    \n",
    "    y, sr = librosa.load(curr_track)\n",
    "    \n",
    "    s_corr = []\n",
    "\n",
    "    for i in range(0,len(y), sr*20):\n",
    "        x = y[i:i+(sr*20)]\n",
    "        i_corr = []\n",
    "        for j in range(0,len(y), sr*20):\n",
    "\n",
    "            if i == j:\n",
    "                pass\n",
    "            else:\n",
    "                yhat = y[j:j+(sr*20)]\n",
    "\n",
    "                corr = np.correlate(np.abs(x), np.abs(yhat))\n",
    "                i_corr.append(corr.mean())\n",
    "\n",
    "        s_corr.append(i_corr)\n",
    "        \n",
    "    \n",
    "    df = pd.DataFrame(s_corr)\n",
    "\n",
    "\n",
    "    plot_df = pd.DataFrame(df.mean())\n",
    "    plot_df.columns = ['corr_mean']\n",
    "    plot_df['section'] = \"Intro\"\n",
    "    section_count = 0\n",
    "\n",
    "    section = 0\n",
    "\n",
    "    i_max = 0\n",
    "\n",
    "    for snippet in range(1,len(plot_df)):\n",
    "\n",
    "        curr_snippet = plot_df.corr_mean[snippet]\n",
    "\n",
    "        if i == 0:\n",
    "            prev_snippet = curr_snippet\n",
    "        else:\n",
    "            prev_snippet = plot_df.corr_mean[snippet-1]  \n",
    "\n",
    "        section_name = \"Section_\" + str(section+1)\n",
    "\n",
    "        if np.abs((curr_snippet - prev_snippet) / prev_snippet) < 0.25:\n",
    "\n",
    "            plot_df.section[snippet] = plot_df.section[snippet-1]\n",
    "\n",
    "        else:\n",
    "\n",
    "            plot_df.section[snippet] = section_name\n",
    "            section+= 1\n",
    "\n",
    "        if snippet == len(plot_df)-1:\n",
    "            plot_df.section[snippet] = \"outro\"\n",
    "\n",
    "        print(np.percentile(plot_df.corr_mean,0.9))\n",
    "        plot_df.section[snippet] = np.where(plot_df.corr_mean[snippet] > np.percentile(plot_df.corr_mean,80), \"Chorus\", plot_df.section[snippet])\n",
    "        \n",
    "        \n",
    "    return plot_df"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
