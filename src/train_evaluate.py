import tensorflow as tf
from multiprocessing import Pool

def evaluate_results(inp_sentence):
  start_token = [len(inp_lang_tokenizer.word_index)+1]
  end_token = [len(inp_lang_tokenizer.word_index)+2]
  
  # inp sentence is the word problem, hence adding the start and end token
  inp_sentence = start_token + list(inp_sentence.numpy()[0]) + end_token
  
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  
  decoder_input = [old_len+1]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == old_len+2:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


def PredictIdx2Word(tensor_idxs, require_idx=False):
  pre_sentence = []
  tensor_idxs = tf.squeeze(tensor_idxs, axis=0)
  for (idx, i) in enumerate(tensor_idxs.numpy()):
    if i in [0, old_len+2] :    # end 
      break  
    if (i < len(targ_lang_tokenizer.word_index) and i not in [0,old_len+1,old_len+2]):
      pre_sentence.append(targ_lang_tokenizer.index_word[i])
  if require_idx:
    return pre_sentence, idx
  else:
    return pre_sentence

def TarIdx2Word(tensor_idxs):
  target_sentence = ''
  tensor_idxs = tf.squeeze(tensor_idxs, axis=0)
  for i in tensor_idxs.numpy():
    if i not in [0,old_len+1,old_len+2]:
      target_sentence += (targ_lang_tokenizer.index_word[i] + ' ')
  target_sentence = target_sentence.split(' ')[:-1]
  return target_sentence

def InpIdx2Sec(tensor_idxs, require_idx=False):
  inp_sentence = ''
  tensor_idxs = tf.squeeze(tensor_idxs, axis=0)
  for (idx, i) in enumerate(tensor_idxs.numpy()):
    if i in [0, inp_len+2]:
      break
    if i not in [0,inp_len+1,inp_len+2]:
      inp_sentence += (inp_lang_tokenizer.index_word[i] + ' ')
  if require_idx:
    return inp_sentence[:-1], idx
  else:
    return inp_sentence

def calAnc(y_true, y_pred):

  for i in range(len(y_true)):
    for j in range(len(y_true[i])):
      y_true[i][j] = targ_lang_tokenizer.word_index[y_true[i][j]]
  for i in range(len(y_pred)):
    for j in range(len(y_pred[i])):
      y_pred[i][j] = targ_lang_tokenizer.word_index[y_pred[i][j]]

  y_true = tf.keras.preprocessing.sequence.pad_sequences(y_true, padding='post', value=3)
  y_pred = tf.keras.preprocessing.sequence.pad_sequences(y_pred, padding='post', value=3)
  # print(y_true[0])
  # print(y_pred[0])
  if y_true.shape[1] < y_pred.shape[1]:
    y_pred = y_pred[:, :y_true.shape[1]]
  else:
    y_true = y_true[:, :y_pred.shape[1]]
  
  table = np.equal(y_true, y_pred)
  # print(table[0])
  table = np.multiply.reduce(table, axis=-1)
  # print(np.count_nonzero(table>0), len(table))

  return np.count_nonzero(table>0), len(table)

def maskValidate(v_data, calLoss=False):
  y_true = []
  y_pred = []
  for (batch, (inp, tar)) in enumerate(v_data):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, _ = transformer(inp, tar_inp, 
                                False, 
                                enc_padding_mask, 
                                combined_mask, 
                                dec_padding_mask)
    if calLoss:
      loss = loss_function(tar_real, predictions)
      val_loss(loss)

    predicted_max = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    list_predicted_max = tf.split(predicted_max, num_or_size_splits=len(predicted_max), axis=0)
    list_target = tf.split(tar, num_or_size_splits=len(tar), axis=0)

    with Pool(processes=8) as pool:
      list_tar_sentence = pool.map(TarIdx2Word, list_target)
      list_predicted_sentence = pool.map(PredictIdx2Word, list_predicted_max)
    y_true.append(list_tar_sentence)
    y_pred.append(list_predicted_sentence)
  y_true = [val for sublist in y_true for val in sublist]
  y_pred = [val for sublist in y_pred for val in sublist]
  return y_true, y_pred

  def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  # train_accuracy(tar_real, predictions)

def train_process(dataset, epoch=-1):
  # 訓練模型 inp -> question, tar -> equation
  dataset.shuffle(BUFFER_SIZE)
  for (batch, (inp, tar)) in enumerate(dataset):
    train_step(inp, tar)
    
    print ('\r', 'Train [{:3d}] Batch [{:4d}] loss {:.4f} perplexity {:.4f}'.format(
        epoch + 1, batch, train_loss.result(), np.exp(train_loss.result())), end='')

def val_process(dataset, dataset_val, epoch=-1, evalue=False):
  
  train_y_true, train_y_pred = maskValidate(dataset)
  val_y_true, val_y_pred = maskValidate(dataset_val, calLoss=True)

  if (evalue == True):
    train_anc, train_total = calAnc(copy.deepcopy(train_y_true), copy.deepcopy(train_y_pred))
    val_anc, val_total = calAnc(copy.deepcopy(val_y_true), copy.deepcopy(val_y_pred))
    if (epoch == -1):
      return val_y_true, val_y_pred, val_anc, val_total
    else:
      print('\nEpoch [{:3d}]  train_loss {:.4f}, train_bleu {:.4f}, train_acc {:.4f}, val_loss {:.4f}, val_bleu {:.4f}, val_acc {:.4f}'.format(
                                                                epoch + 1,
                                                                float(train_loss.result()),
                                                                corpus_bleu(train_y_true, train_y_pred, smoothing_function=smoothie.method4),
                                                                float(train_anc)/train_total,
                                                                float(val_loss.result()),
                                                                corpus_bleu(val_y_true, val_y_pred, smoothing_function=smoothie.method4),
                                                                float(val_anc)/val_total,
                                                                ))
      return float(val_anc)/val_total
  elif not evalue:
    print('\nEpoch [{:3d}]             train_loss {:.4f}, val_loss {:.4f}'.format(epoch + 1, float(train_loss.result()), float(val_loss.result())))


def evaluate(inp_sentence):
  start_token = [len(inp_lang_tokenizer.word_index)+1]
  end_token = [len(inp_lang_tokenizer.word_index)+2]
  
  # inp sentence is the word problem, hence adding the start and end token
  inp_sentence = start_token + [inp_lang_tokenizer.word_index[i] for i in preprocess_input(inp_sentence).split(' ')]+end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # start with equation's start token
  decoder_input = [old_len+1]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :] 
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == old_len+2:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)
  return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer, idx=-1, input_idx=-1):
  fig = plt.figure(figsize=(24, 12))
  sentence = preprocess_input(sentence)
  attention = tf.squeeze(attention[layer], axis=0)
  if idx!=-1:
    print('idx: {}, input_idx: {}'.format(idx, input_idx), '  attention.shape: ', attention[0][:idx, :input_idx+1].shape)
  
  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # plot the attention weights
    if input_idx != -1:
      ax.matshow(attention[head][:idx, :input_idx+1], cmap='viridis')
    else:
      ax.matshow(attention[head][:idx, :], cmap='viridis')
    
    fontdict = {'fontsize': 12}
    
    ax.set_xticks(range(len(sentence.split(' '))+2))

    align_seq = []
    for i in list(result.numpy()):
      if i in [0, old_len+2]:
        break
      if i < len(targ_lang_tokenizer.word_index) and i not in [0,old_len+1,old_len+2]:
        align_seq.append(targ_lang_tokenizer.index_word[i])

    ax.set_yticks(range(len(align_seq)+3))
    
    
    ax.set_ylim(len(align_seq), -0.5)

    ax.set_xticklabels(
        ['<start>']+sentence.split(' ')+['<end>'], 
        fontdict=fontdict, rotation=90)
    
    ax.set_yticklabels(align_seq, fontdict=fontdict)
    
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show()


def translate(sentence, plot=''):
  result, attention_weights = evaluate(sentence)
  # print('result',list(result.numpy()))

  # use the result tokens to convert prediction into a list of characters
  # (not inclusing padding, start and end tokens)
  predicted_sentence = TarIdx2Word(tf.expand_dims(result, axis=0))  
  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(' '.join(predicted_sentence)))
  
  if plot:
    plot_attention_weights(attention_weights, sentence, result, plot)


