import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser(description='pre process data')
parser.add_argument('--input-images', type=str, default='', help='input images')
parser.add_argument('--output-images', type=str, default='', help='output images')
args = parser.parse_args()


def main():
    os.makedirs(args.output_images, exist_ok=True)
    #folders = ['African', 'Caucasian', 'Asian', 'Indian']
    folders = ['Asian']
    total_person_id = 0
    for folder in folders:
        data_path = os.path.join(args.input_images, folder)
        data_list = os.listdir(data_path)
        data_len = len(data_list)
        print('{} has {} people.'.format(folder, data_len))

        if folder == 'Caucasian':
            val_num = int(data_len * 0.0)
        else:
            val_num = int(data_len * 0.0)

        val_list = random.sample(data_list, val_num)
        train_list = list(set(data_list).difference(set(val_list)))

        for person in train_list:
            person_path = os.path.join(args.input_images, folder, person)
            for index, single_img in enumerate(os.listdir(person_path)):
                old_img = os.path.join(person_path, single_img)
                new_img = os.path.join(args.output_images, str(total_person_id) + '_' + str(index) + '.jpg')
                shutil.copyfile(old_img, new_img)
            total_person_id += 1


if __name__ == '__main__':
    main()
