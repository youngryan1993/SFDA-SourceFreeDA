from data import *
from lib import  *
import time

def APM_init_update(feature_extractor, classifier_t):
    start_time = time.time()
    available_cls = []
    h_dict = {}
    feat_dict = {}
    missing_cls = []
    after_softmax_numpy_for_emergency = []
    feature_numpy_for_emergency = []
    max_prototype_bound = 100

    for cls in range(len(source_classes)):
        h_dict[cls] = []
        feat_dict[cls] = []

    for (im_target_lbcorr, label_target_lbcorr) in target_train_dl:
        im_target_lbcorr = im_target_lbcorr.cuda()
        fc1_lbcorr = feature_extractor.forward(im_target_lbcorr)
        _, _, _, after_softmax = classifier_t.forward(fc1_lbcorr)
        after_softmax_numpy_for_emergency.append(after_softmax.data.cpu().numpy())
        feature_numpy_for_emergency.append(fc1_lbcorr.data.cpu().numpy())

        pseudo_label = torch.argmax(after_softmax, dim=1)
        pseudo_label = pseudo_label.cpu()

        entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
        entropy_norm = entropy / np.log(after_softmax.size(1))
        entropy_norm = entropy_norm.squeeze(1)
        entropy_norm = entropy_norm.cpu()

        for cls in range(len(source_classes)):
            # stack H for each class
            cls_filter = (pseudo_label == cls)
            list_loc = (torch.where(cls_filter == 1))[0]
            num_element = list(list_loc.data.numpy())
            if len(list_loc) == 0:
                missing_cls.append(cls)
                continue
            available_cls.append(cls)
            filtered_ent = torch.gather(entropy_norm, dim=0, index=list_loc)
            filtered_feat = torch.gather(fc1_lbcorr.cpu(), dim=0, index=list_loc.unsqueeze(1).repeat(1, 2048))

            h_dict[cls].append(filtered_ent.cpu().data.numpy())
            feat_dict[cls].append(filtered_feat.cpu().data.numpy())

    available_cls = np.unique(available_cls)

    prototype_memory = []
    prototype_memory_dict = {}
    after_softmax_numpy_for_emergency = np.concatenate(after_softmax_numpy_for_emergency, axis=0)
    feature_numpy_for_emergency = np.concatenate(feature_numpy_for_emergency, axis=0)

    max_top1_ent = 0
    for cls in available_cls:
        ents_np = np.concatenate(h_dict[cls], axis=0)
        ent_idxs = np.argsort(ents_np)
        top1_ent = ents_np[ent_idxs[0]]
        if max_top1_ent < top1_ent:
            max_top1_ent = top1_ent
            max_top1_class = cls

    class_protypeNum_dict = {}
    max_prototype = 0

    for cls in available_cls:
        ents_np = np.concatenate(h_dict[cls], axis=0)
        ents_np_filtered = (ents_np <= max_top1_ent)
        class_protypeNum_dict[cls] = ents_np_filtered.sum()

        if max_prototype < ents_np_filtered.sum():
            max_prototype = ents_np_filtered.sum()

    if max_prototype > 100:
        max_prototype = max_prototype_bound

    for cls in range(len(source_classes)):

        if cls in available_cls:
            ents_np = np.concatenate(h_dict[cls], axis=0)
            feats_np = np.concatenate(feat_dict[cls], axis=0)
            ent_idxs = np.argsort(ents_np)

            truncated_feat = feats_np[ent_idxs[:class_protypeNum_dict[cls]]]
            fit_to_max_prototype = np.concatenate([truncated_feat] * (int(max_prototype / truncated_feat.shape[0]) + 1),
                                                  axis=0)
            fit_to_max_prototype = fit_to_max_prototype[:max_prototype, :]

            prototype_memory.append(fit_to_max_prototype)
            prototype_memory_dict[cls] = fit_to_max_prototype
        else:
            after_softmax_torch_for_emergency = torch.Tensor(after_softmax_numpy_for_emergency)
            emergency_idx = torch.argsort(after_softmax_torch_for_emergency, descending=True, dim=1)
            cls_emergency_idx = emergency_idx[:, cls]
            cls_emergency_idx = cls_emergency_idx[0]
            cls_emergency_idx_numpy = cls_emergency_idx.data.numpy()

            copied_features_emergency = np.concatenate(
                [np.expand_dims(feature_numpy_for_emergency[cls_emergency_idx_numpy], axis=0)] * max_prototype, axis=0)

            prototype_memory.append(copied_features_emergency)
            prototype_memory_dict[cls] = copied_features_emergency

    print("** APM update... time:", time.time() - start_time)
    prototype_memory = np.concatenate(prototype_memory, axis=0)
    num_prototype_ = int(max_prototype)

    return prototype_memory, num_prototype_, prototype_memory_dict

