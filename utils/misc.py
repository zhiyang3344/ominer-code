

def sample_palce365():
    from shutil import copyfile
    import os
    src_path='../../datasets/ood_datasets/test_256'
    dst_path='../../datasets/ood_datasets/places365/test_subset/test_subset'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    f = open('../datasets/ood_datasets/places365/places365_test_list.txt', 'r')
    count=1
    for line in f:
        src_file=os.path.join(src_path, line.strip())
        dst_file=os.path.join(dst_path, line.strip())
        copyfile(src_file, dst_file)
        print('{} copy file from {} to {}'.format(count, src_file, dst_file))
        count+=1


if __name__ == '__main__':
    sample_palce365()