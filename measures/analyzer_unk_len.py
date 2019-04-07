import sys
# from measures import evaluation_utils
# from measures import embedding_metrics

    

if __name__ == "__main__":
    filename = sys.argv[1]
    # references = []
    # predictions = []
    # modelnm = ""
    print("filen name:", filename)
    unk_cnt = 0
    word_cnt = 0
    res_len = 0
    res_cnt = 0
    with open(filename) as f:
        for idx, line in enumerate(f):
            if line.strip() == "": continue
            # if line.startswith("ref: "):
            #     references.append(line[len("ref: "):])
            if line.startswith("res: "):
                # end_index = line.find(" _EOS ")
                # if end_index != -1:
                #     line = line[:end_index] + "\n"
                # predictions.append(line[len("res: "):])
                word_array = line[len("res: "):].split(" ")
                for word in word_array:
                    word_cnt += 1
                    if word == 'UNK': unk_cnt += 1
                res_len += len("".join(word_array))
                res_cnt += 1
    print("avg. len:", res_len/res_cnt)
    print("unk rate:", unk_cnt/word_cnt)
            
        # if len(references) != 0:
        #     assert len(references) == len(predictions)
        #     print(modelnm)
        #     write_files(references, predictions)
        #     evaluation()
        #     references = []
        #     predictions = []
