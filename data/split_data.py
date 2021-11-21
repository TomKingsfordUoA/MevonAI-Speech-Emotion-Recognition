import glob
import pathlib
import os

from sklearn.model_selection import train_test_split
import pandas as pd


class Ravdess:
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '07': 'disgust',
        '08': 'ps',
    }

    channels = {
        '01': 'speech',
        '02': 'song',
    }

    modalities = {
        '01': 'full-AV',
        '02': 'video-only',
        '03': 'audio-only',
    }

    @staticmethod
    def parse_ravdess_filename(filename: str) -> dict:
        name, suffix = filename.split('.')
        assert suffix.lower() in {'mp4', 'wav'}
        modality, channel, emotion, intensity, statement, repetition, actor = name.split('-')
        return {
            'modality': Ravdess.modalities[modality],
            'channel': Ravdess.channels[channel],
            'emotion': Ravdess.emotions[emotion],
            'intensity': intensity,
            'statement': statement,
            'repetition': repetition,
            'actor': actor,
        }


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_dir, 'data')

# Get all filepaths in datasets:
ravdess_paths = [(path, Ravdess.parse_ravdess_filename(os.path.basename(path))) for path in glob.glob(os.path.join(data_dir, 'RAVDESS_extracted/*/*.mp4'))]
ravdess_paths.extend([(path, Ravdess.parse_ravdess_filename(os.path.basename(path))) for path in glob.glob(os.path.join(data_dir, 'RAVDESS_extracted/*/*.wav'))])
# tess_paths_all = glob.glob(os.path.join(data_dir, 'TESS/*.wav'))  # TODO(TK): enable TESS

# Filter acceptable instances (e.g. remove RAVDESS singing)
ravdess_paths = [
    (path, metadata)
    for path, metadata in ravdess_paths
    if metadata['modality'] == 'audio-only' and metadata['channel'] == 'speech'
]

# Extract emotions:
ravdess_paths_with_emotion = [{'path': pathlib.Path(path).relative_to(project_dir), 'emotion': metadata['emotion']} for path, metadata in ravdess_paths]

# split the files into train and test, by actor:
ravdess_actors = list({metadata['actor'] for _, metadata in ravdess_paths})
train_ravdess_actors, test_ravdess_actors = train_test_split(ravdess_actors, test_size=0.2, random_state=42)
idx_ravdess_train = [idx for idx, (path, metadata) in enumerate(ravdess_paths) if metadata['actor'] in train_ravdess_actors]
idx_ravdess_test = [idx for idx, (path, metadata) in enumerate(ravdess_paths) if metadata['actor'] in test_ravdess_actors]
df_ravdess = pd.DataFrame(ravdess_paths_with_emotion)
df_ravdess_train = df_ravdess.iloc[idx_ravdess_train].sort_index()
df_ravdess_test = df_ravdess.iloc[idx_ravdess_test].sort_index()

print(df_ravdess.shape)
print(df_ravdess_train.shape)
print(df_ravdess_test.shape)

print(df_ravdess.head())
print(df_ravdess_train.head())
print(df_ravdess_test.head())

# Write to csv
df_ravdess_train.to_csv(os.path.join(data_dir, 'train_ravdess.csv'), index=False, header=True)
df_ravdess_test.to_csv(os.path.join(data_dir, 'test_ravdess.csv'), index=False, header=True)

