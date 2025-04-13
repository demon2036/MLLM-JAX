from datasets import load_dataset


dataset=load_dataset('YouJiacheng/DAPO-Math-17k-dedup',split='train')


for data in dataset:
    print(data)
    break

