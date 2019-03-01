import sys
from measures import evaluation_utils
# from measures import embedding_metrics

reference_file_path = "/home/renxinzhang/renxingzhang/tmp/references.txt"
prediction_file_path = "/home/renxinzhang/renxingzhang/tmp/predictions.txt"

def write_files(references, predictions):
    with open(reference_file_path, "w") as f:
        f.writelines(references)
    with open(prediction_file_path, "w") as f:
        f.writelines(predictions)

def evaluation():
    from nlgeval import compute_metrics
    metrics_dict = compute_metrics(hypothesis=prediction_file_path,
                                references=[reference_file_path])
    print(metrics_dict)

    from measures import selfbleu
    selfbleuobj = selfbleu.SelfBleu(prediction_file_path, 1)
    print("selfbleu-1", selfbleuobj.get_score())
    selfbleuobj = selfbleu.SelfBleu(prediction_file_path, 2)
    print("selfbleu-2", selfbleuobj.get_score())

    # embedding_metrics.metrics_embeddings(reference_file_path,
    #     prediction_file_path)

    eval_log = {}
    for metric in ['bleu','rouge','accuracy','word_accuracy']:
        score = evaluation_utils.evaluate(
            reference_file_path,
            prediction_file_path,
            metric)
        eval_log[metric] = score
        if metric == "bleu":
            print("  bleu-1, bleu-2, bleu-3, bleu-4: %.5f,  %.5f,  %.5f,  %.5f" % score)
        elif metric == "rouge":
            print("  rouge-1, rouge-2, rouge-l: %.5f,  %.5f,  %.5f" % score)
        else:
            print("  %s: %.5f" % (metric, score))
    

if __name__ == "__main__":
    filename = sys.argv[1]
    references = []
    predictions = []
    modelnm = ""
    print("filen name:", filename)
    with open(filename) as f:
        for idx, line in enumerate(f):
            if line.strip() == "": continue
            if line.startswith("ref: "):
                references.append(line[len("ref: "):])
            if line.startswith("res: "):
                end_index = line.find(" _EOS ")
                if end_index != -1:
                    line = line[:end_index] + "\n"
                predictions.append(line[len("res: "):])
            
        if len(references) != 0:
            assert len(references) == len(predictions)
            print(modelnm)
            write_files(references, predictions)
            evaluation()
            references = []
            predictions = []
