import os
import shutil

def checkpoint(isTrain = False):
  if isTrain:
    shutil.rmtree(os.path.join(drive_root, "ADL Project")
  checkpoint_dir = os.path.join(drive_root, "ADL Project/checkpoints")
  checkpoint_dir = os.path.join(checkpoint_dir, "training_checkpoints/akshata_transfomer")

  print("Checkpoints directory is", checkpoint_dir)
  if os.path.exists(checkpoint_dir):
    print("Checkpoints folder already exists")
  else:
    print("Creating a checkpoints directory")
  os.makedirs(checkpoint_dir)


  checkpoint = tf.train.Checkpoint(transformer=transformer,
                          optimizer=optimizer)

  ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
  latest = ckpt_manager.latest_checkpoint
  if latest:
    print("Model exists")
    checkpoint.restore(latest)
  else:
    print("Model doesn't exists")
    exit(-1)
