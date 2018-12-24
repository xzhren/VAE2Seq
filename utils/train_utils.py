
def show_loss(x_log, y_log, t_log, log):
    if log:
        print(" | merged_loss:%.1f | trans_w:%.3f | encoder_loss:%.1f | decoder_loss:%.1f" % (log['merged_loss'], log['trans_loss'], log['encoder_loss'], log['decoder_loss']))
    if x_log:
        print(" | x_loss:%1f | x_nll_loss:%.1f | x_kl_w:%.3f | x_kl_loss:%.2f" % (x_log['loss'], x_log['nll_loss'], x_log['kl_w'], x_log['kl_loss']))
    if y_log:
        print(" | y_loss:%1f | y_nll_loss:%.1f | y_kl_w:%.3f | y_kl_loss:%.2f" % (y_log['loss'], y_log['nll_loss'], y_log['kl_w'], y_log['kl_loss']))
    if t_log:
        print(" | trans_loss:%.6f" % (t_log['trans_loss']))

def summary_flush(x_log, y_log, t_log, log, summary_writer):
    for log in [x_log, y_log, t_log, log]:
        if log:
            summaries, train_step = log['summaries'], log['step']
            summary_writer.add_summary(summaries, train_step) # write the summaries
            
            if train_step % 100 == 0: # flush the summary writer every so often
                summary_writer.flush()
    return train_step


