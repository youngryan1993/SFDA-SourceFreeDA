from data import *
from net import *
from lib import  *
from torch import optim
from APM_update import *
import torch.backends.cudnn as cudnn
import time

cudnn.benchmark = True
cudnn.deterministic = True

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

save_model_path = 'pretrained_weights/'+str(args.data.dataset.source)+str(args.data.dataset.target)+'/'+'domain'+ str(args.data.dataset.source)+str(args.data.dataset.target)+'accBEST_model_checkpoint.pth.tar'
save_model_statedict = torch.load(save_model_path)['state_dict']

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}

# ======= network architecture =======
class Target_TrainableNet(nn.Module):
    def __init__(self):
        super(Target_TrainableNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.cls_multibranch = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)


# ======= target network =======
trainable_tragetNet = Target_TrainableNet()
trainable_tragetNet.load_state_dict(save_model_statedict)

feature_extractor_t =(trainable_tragetNet.feature_extractor).cuda()
classifier_s2t = (trainable_tragetNet.classifier).cuda()
classifier_t = (trainable_tragetNet.cls_multibranch).cuda()
print ("Finish model loaded...")


domains=['amazon', 'dslr', 'webcam']
print ('domain....'+domains[args.data.dataset.source]+'>>>>>>'+domains[args.data.dataset.target])

counter = AccuracyCounter()
with TrainingModeManager([feature_extractor_t, classifier_t], train=False) as mgr, torch.no_grad():

        for i, (img, label) in enumerate(target_test_dl):
            img = img.cuda()
            label = label.cuda()

            feature = feature_extractor_t.forward(img)
            ___, __, before_softmax, predict_prob = classifier_t.forward(feature)

            counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

        acc_test = counter.reportAccuracy()
        print('>>>>>>>Test Accuracy>>>>>>>>>>.')
        print(acc_test)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')

exit()