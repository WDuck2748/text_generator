#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder
#from text_generator import model
#from text_generator import sample
#from text_generator import encoder


def generate_text(imput_text):
    model_name='355M'
    seed=None
    nsamples=1
    batch_size=1
    length=100
    temperature=1
    top_k=40
    top_p=0.9

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    cur_path = os.getcwd() + "/models" + "/" + model_name
    print(cur_path)
    with open(cur_path + '/hparams.json') as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(cur_path))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(imput_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
    return text

text = generate_text("How are you doing today?")
print(text)

WriteTxtFile = open("write-demo.txt", "w")
WriteTxtFile.write(text)
WriteTxtFile.close()