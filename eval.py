from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import numpy as np
import matplotlib.pyplot as plt
torch.nn.Module.dump_patches=True
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = 'Conversion/Root/Data/Output'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 4
workers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
#model.eval()

# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)



def savbh(x, y, l):
    fig = plt.figure(figsize=(30,100))
    plt.barh(x, y)
    plt.title('Different mAPs for different combinations')
    plt.xlabel('mAPs')
    plt.ylabel('Different Combinations')
    plt.savefig(str(l)+'BAR_EVAL_RESULTS.png',dpi=fig.dpi)

def savl(x, y, l):
    fig = plt.figure(figsize=(30,30))
    plt.plot(y)
    plt.title('Different mAPs for different combinations')
    plt.xlabel('Different Combinations')
    plt.ylabel('mAPs')
    plt.savefig(str(l)+'LINE_EVAL_RESULTS.png',dpi=fig.dpi)

def savl1(x, y, l):
    fig = plt.figure(figsize=(30,30))
    plt.plot(x,y)
    plt.title('Different mAPs for different combinations')
    plt.xlabel('Different Combinations')
    plt.ylabel('mAPs')
    plt.xticks(x, rotation=90)
    plt.savefig(str(l)+'LINE1_EVAL_RESULTS.png',dpi=fig.dpi)

def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """


    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Make sure it's in eval mode
        model.eval()
        scrs = [0.1, 0.15, 0.2, 0.25, 0.3]
        over = [0.5, 0.6, 0.7]
        topks = [2, 4, 6, 8, 10, 12]
        combinations = [] 
        maps=[]
        als=[]
        for s in range(len(scrs)):
          for o in range(len(over)):
            for k in range(len(topks)):   
              for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
                  images = images.to(device)  # (N, 3, 300, 300)

                  # Forward prop.
                  predicted_locs, predicted_scores = model(images)
                  
                  # Detect objects in SSD output
                  det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                            min_score=scrs[s], max_overlap=over[o],
                                                                                            top_k=topks[k])
                  # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

                  # Store this batch's results for mAP calculation
                  boxes = [b.to(device) for b in boxes]
                  labels = [l.to(device) for l in labels]
                  difficulties = [d.to(device) for d in difficulties]

                  det_boxes.extend(det_boxes_batch)
                  det_labels.extend(det_labels_batch)
                  det_scores.extend(det_scores_batch)
                  true_boxes.extend(boxes)
                  true_labels.extend(labels)
                  true_difficulties.extend(difficulties)

              # Calculate mAP
              APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
              print('')
              print('Eval with min_score as: ', scrs[s],' max_overlap as: ',over[o],' and top_k as: ',topks[k])
              temp = 's: ' + str(scrs[s]) + ' o: ' + str(over[o])+ ' k: ' + str(topks[k])
              combinations.append(temp)
              # Print AP for each class
              pp.pprint(APs)
              maps.append(mAP)
              als.append(APs)
              print('Mean Average Precision (mAP): %.3f' % mAP)
              print('')
              print('Before len: ',len(det_boxes), len(det_labels), len(det_scores), len(true_boxes), len(true_labels), len(true_difficulties))
              det_boxes.clear(), det_labels.clear(), det_scores.clear(), true_boxes.clear(), true_labels.clear(), true_difficulties.clear()
              print('After len: ',len(det_boxes), len(det_labels), len(det_scores), len(true_boxes), len(true_labels), len(true_difficulties))
    print('')
    print('maps: ',maps)
    print('')
    print('als: ',als)
    print('')
    print('All combinations: ',combinations)
    (m,i) = max((v,i) for i,v in enumerate(maps))
    print('')
    print('The max map we got is: ',m,' with the combination as: ',combinations[i])
    savbh(combinations, maps, 1)
    savl(combinations, maps, 1)
    savl1(combinations, maps, 1)
if __name__ == '__main__':
    evaluate(test_loader, model)
