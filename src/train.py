from data_inputs import *
import numpy as np
import models
import time


def main():
    low_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHENNELS])
    high_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, LABEL_SIZE, LABEL_SIZE, NUM_CHENNELS])

    inferences = models.create_model(MODEL_NAME, low_res_holder)
    training_loss = models.loss(inferences, high_res_holder, name='training_loss', weights_decay=0)
    validation_loss = models.loss(inferences, high_res_holder, name='validation_loss')
    tf.scalar_summary('training_loss', training_loss)
    tf.scalar_summary('validation_loss', validation_loss)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    # learning_rate = tf.train.piecewise_constant(
    #     global_step,
    #     [2000, 5000, 8000, 12000, 16000],
    #     [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
    # )
    learning_rate = tf.train.inverse_time_decay(0.001, global_step, 10000, 2)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(training_loss, global_step=global_step)

    low_res_batch, high_res_batch = batch_queue_for_training(TRAINING_DATA_PATH)
    low_res_eval, high_res_eval = batch_queue_for_testing(VALIDATION_DATA_PATH)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Start the queue runners (make batches).
    tf.train.start_queue_runners(sess=sess)

    # the saver will restore all model's variables during training
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=MAX_CKPT_TO_KEEP)
    # Merge all the summaries and write them out to TRAINING_DIR
    merged_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(TRAINING_SUMMARY_PATH, sess.graph)

    for step in range(1, NUM_TRAINING_STEPS+1):
        start_time = time.time()
        low_res_images, high_res_images = sess.run([low_res_batch, high_res_batch])
        feed_dict = {low_res_holder: low_res_images, high_res_holder: high_res_images}
        _, batch_loss = sess.run([train_step, training_loss], feed_dict=feed_dict)
        duration = time.time() - start_time
        assert not np.isnan(batch_loss), 'Model diverged with loss = NaN'

        if step % 100 == 0:  # show training status
            num_examples_per_step = BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = 'step %d, batch_loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (step, batch_loss, examples_per_sec, sec_per_batch))

        if step % 1000 == 0:  # run validation and show its result
            low_res_images, high_res_images = sess.run([low_res_eval, high_res_eval])
            feed_dict = {low_res_holder: low_res_images, high_res_holder: high_res_images}
            batch_loss = sess.run(validation_loss, feed_dict=feed_dict)
            print('step %d, validation loss = %.3f' % (step, batch_loss))

            summary = sess.run(merged_summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary)

        # Save the model checkpoint periodically.
        if step % 10000 == 0 or (step + 1) == NUM_TRAINING_STEPS:
            saver.save(sess, join(CHECKPOINTS_PATH, 'model.ckpt'), global_step=step)

    print('Training Finished!')


if __name__ == '__main__':
    main()
