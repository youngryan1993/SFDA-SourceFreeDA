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

save_model_path = 'source_pretrained_weights/'+ str(args.data.dataset.source)+str(args.data.dataset.target)+'/'+'model_checkpoint.pth.tar'
save_model_statedict = torch.load(save_model_path)['state_dict']

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}


# ===  Network structure =================
class Source_FixedNet(nn.Module):
    def __init__(self):
        super(Source_FixedNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)

class Target_TrainableNet(nn.Module):
    def __init__(self):
        super(Target_TrainableNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.cls_multibranch = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)


# === source pretrained network ====================
fixed_sourceNet = Source_FixedNet()
fixed_sourceNet.load_state_dict(save_model_statedict)
fixed_feature_extractor =(fixed_sourceNet.feature_extractor).cuda()
fixed_classifier = (fixed_sourceNet.classifier).cuda()
fixed_feature_extractor.eval()
fixed_classifier.eval()

# === target network   =================
trainable_tragetNet = Target_TrainableNet()
feature_extractor =(trainable_tragetNet.feature_extractor).cuda()
feature_extractor.load_state_dict(fixed_sourceNet.feature_extractor.state_dict())
classifier_s2t = (trainable_tragetNet.classifier).cuda()
classifier_s2t.load_state_dict(fixed_sourceNet.classifier.state_dict())
classifier_t = (trainable_tragetNet.cls_multibranch).cuda()
classifier_t.load_state_dict(fixed_sourceNet.classifier.state_dict())


model_dict = {
            'global_step':0,
            'state_dict': trainable_tragetNet.state_dict(),
            'accuracy': 0}


feature_extractor.train()
classifier_s2t.train()
classifier_t.train()

print ("Finish model loaded...")



domains=['amazon', 'dslr', 'webcam']
print ('domain....'+domains[args.data.dataset.source]+'>>>>>>'+domains[args.data.dataset.target])


scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=(args.train.min_step))

optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_classifier_s2t = OptimWithSheduler(
    optim.SGD(classifier_s2t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_classifier_t= OptimWithSheduler(
    optim.SGD(classifier_t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)

global_step = 0
best_acc = 0
epoch_id = 0
class_num =  args.data.dataset.n_total
pt_memory_update_frequncy =  args.train.update_freq


while global_step < args.train.min_step:

    epoch_id += 1

    for i, (im_target, label_target) in enumerate(target_train_dl):

        if (global_step) % pt_memory_update_frequncy == 0:
            # TODO ==== APM init/update
            prototype_memory, num_prototype_,prototype_memory_dict = APM_init_update(feature_extractor, classifier_t)

        im_target = im_target.cuda()

        # TODO ==== forward pass:  source-pretrained networks
        fixed_fc1_t = fixed_feature_extractor.forward(im_target)
        _, _, _, pseudo_logit_target = fixed_classifier.forward(fixed_fc1_t)
        pseudo_label_source = torch.argmax(pseudo_logit_target, dim=1)

        # TODO ==== forward pass:  target networks
        fc1_t = feature_extractor.forward(im_target)
        fc1_s, feature_bottleneck, fc2_t, predict_prob_target = classifier_s2t.forward(fc1_t)
        _, _, multibranch_logit, _  = classifier_t(fc1_t)

        # TODO ==== Compute pesudo label
        proto_feat_tensor = torch.Tensor(prototype_memory) # (B * 2048)
        feature_embed_tensor = fc1_t.cpu()
        proto_feat_tensor = tensor_l2normalization(proto_feat_tensor)
        batch_feat_tensor = tensor_l2normalization(feature_embed_tensor)

        sim_mat = torch.mm(batch_feat_tensor, proto_feat_tensor.permute(1,0))
        sim_mat = F.avg_pool1d(sim_mat.unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0)# (B, #class)
        pseudo_label_corrected = torch.argmax(sim_mat, dim=1).cuda()

        # TODO ==== Confidence-based filtering
        arg_idxs = torch.argsort(sim_mat, dim=1, descending=True) # (B, #class)

        first_group_idx = arg_idxs[:, 0]
        second_group_idx = arg_idxs[:, 1]

        first_group_feat = [prototype_memory_dict[int(x.data.numpy())] for x in first_group_idx]
        first_group_feat_tensor = torch.tensor(np.concatenate(first_group_feat, axis=0)) # (B*P, 2048)
        first_group_feat_tensor = tensor_l2normalization(first_group_feat_tensor)

        second_group_feat = [prototype_memory_dict[int(x.data.numpy())] for x in second_group_idx]
        second_group_feat_tensor = torch.tensor(np.concatenate(second_group_feat, axis=0)) # (B*P, 2048)
        second_group_feat_tensor = tensor_l2normalization(second_group_feat_tensor)

        feature_embed_tensor_repeat = torch.Tensor(np.repeat(feature_embed_tensor.cpu().data.numpy(), repeats=num_prototype_, axis=0))
        feature_embed_tensor_repeat = tensor_l2normalization(feature_embed_tensor_repeat)

        first_dist_mat = 1 - torch.mm(first_group_feat_tensor, feature_embed_tensor_repeat.permute(1,0)) # distance = 1  - simialirty
        second_dist_mat = 1 - torch.mm(second_group_feat_tensor, feature_embed_tensor_repeat.permute(1,0))

        first_dist_mat = F.max_pool2d(first_dist_mat.permute(1,0).unsqueeze(0).unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0).squeeze(0)# (B, #class)
        second_dist_mat = -1*F.max_pool2d(-1* second_dist_mat.permute(1,0).unsqueeze(0).unsqueeze(0), kernel_size=num_prototype_, stride=num_prototype_).squeeze(0).squeeze(0)# (B, #class)

        first_dist_vec = torch.diag(first_dist_mat) #(B)
        second_dist_vec = torch.diag(second_dist_mat) # B

        confidence_mask = ((first_dist_vec- second_dist_vec) < 0).cuda()

        # TODO ===== Optimize networks based on the combination of two losses

        alpha = np.float(2.0 / (1.0 + np.exp(-10 * global_step / float(args.train.min_step//2))) - 1.0)
        ce_from_sorucenet = nn.CrossEntropyLoss()(fc2_t, pseudo_label_source)
        ce_from_confidence = nn.CrossEntropyLoss(reduction='none')(multibranch_logit, pseudo_label_corrected).view(-1, 1).squeeze(1)
        ce_from_confidence = torch.mean(ce_from_confidence * confidence_mask, dim=0, keepdim=True)

        ce = (1-alpha) * ce_from_sorucenet + alpha * ce_from_confidence

        with OptimizerManager([optimizer_finetune, optimizer_classifier_s2t, optimizer_classifier_t]):
            loss = ce
            loss.backward()

        global_step += 1

        # TODO ===== Evaluation during training
        if global_step % args.test.test_interval == 0:

            counter = AccuracyCounter()
            with TrainingModeManager([feature_extractor, classifier_t], train=False) as mgr, torch.no_grad():

                for i, (im, label) in enumerate(target_test_dl):
                    im = im.cuda()
                    label = label.cuda()

                    feature = feature_extractor.forward(im)
                    ___, __, before_softmax, predict_prob = classifier_t.forward(feature)

                    counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

            acc_test = counter.reportAccuracy()
            print('>>>>>>>>>>>accuracy>>>>>>>>>>>>>>>>.')
            print(acc_test)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')
            if best_acc < acc_test:
                best_acc = acc_test
                model_dict = {
                        'global_step': global_step + 1,
                        'state_dict': trainable_tragetNet.state_dict(),
                        'accuracy': acc_test}

                torch.save(model_dict, join('pretrained_weights/'+str(args.data.dataset.source) + str(args.data.dataset.target) +'/' + 'domain'+ str(args.data.dataset.source)+str(args.data.dataset.target)+'accBEST_model_checkpoint.pth.tar'))



counter = AccuracyCounter()
with TrainingModeManager([feature_extractor, classifier_t], train=False) as mgr, torch.no_grad():

        for i, (im, label) in enumerate(target_test_dl):
            im = im.cuda()
            label = label.cuda()

            feature = feature_extractor.forward(im)
            ___, __, before_softmax, predict_prob = classifier_t.forward(feature)

            counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

        acc_test = counter.reportAccuracy()
        print('>>>>>>>Final accuracy>>>>>>>>>>.')
        print(acc_test)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.')

        if best_acc < acc_test:
            best_acc = acc_test
            model_dict = {
                'global_step': global_step + 1,
                'state_dict': trainable_tragetNet.state_dict(),
                'accuracy': acc_test}

            torch.save(model_dict, join('pretrained_weights/'+str(args.data.dataset.source) + str(
                args.data.dataset.target) +'/' + 'domain' + str(args.data.dataset.source) + str(
                args.data.dataset.target) + 'accBEST_model_checkpoint.pth.tar'))


exit()