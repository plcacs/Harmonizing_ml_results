    def forward(self, images: FloatTensor, sizes: FloatTensor, image_features: List[Tensor]) -> RegionDetectorOutput:
        raise NotImplementedError()

    def forward(self, images: FloatTensor, sizes: FloatTensor, image_features: List[Tensor]) -> RegionDetectorOutput:
        batch_size, num_features, height, width = images.size()
        features: List[Tensor] = [self._seeded_random_tensor(2, 10, device=images.device) for _ in range(batch_size)]
        boxes: List[Tensor] = [torch.zeros(2, 4, dtype=torch.float32, device=images.device) for _ in range(batch_size)]
        for image_num in range(batch_size):
            boxes[image_num][0, 2] = sizes[image_num, 0]
            boxes[image_num][0, 3] = sizes[image_num, 1]
            boxes[image_num][1, 2] = sizes[image_num, 0]
            boxes[image_num][1, 3] = sizes[image_num, 1]
        return RegionDetectorOutput(features, boxes)

    def forward(self, images: FloatTensor, sizes: FloatTensor, image_features: List[Tensor]) -> RegionDetectorOutput:
        if self.training:
            raise RuntimeError('FasterRcnnRegionDetector can not be used for training at the moment')
        image_shapes = list(((int(h), int(w)) for h, w in sizes))
        image_list = torchvision.models.detection.image_list.ImageList(images, image_shapes)
        proposals, _ = self.detector.rpn(image_list, image_features)
        box_features = self.detector.roi_heads.box_roi_pool(image_features, proposals, image_shapes)
        box_features = self.detector.roi_heads.box_head(box_features)
        class_logits, box_regression = self.detector.roi_heads.box_predictor(box_features)
        boxes, features, scores, labels = self._postprocess_detections(class_logits, box_features, box_regression, proposals, image_shapes)
        return RegionDetectorOutput(features, boxes, scores, labels)

    def _postprocess_detections(self, class_logits: Tensor, box_features: Tensor, box_regression: Tensor, proposals: List[Tensor], image_shapes: List[Tuple[int, int]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.detector.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        features_list = box_features.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        all_boxes = []
        all_features = []
        all_scores = []
        all_labels = []
        for boxes, features, scores, image_shape in zip(pred_boxes_list, features_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            features = features.unsqueeze(1).expand(boxes.shape[0], boxes.shape[1], -1)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = boxes[:, 1:]
            features = features[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            boxes = boxes.reshape(-1, 4)
            features = features.reshape(boxes.shape[0], -1)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            inds = torch.where(scores > self.detector.roi_heads.score_thresh)[0]
            boxes, features, scores, labels = (boxes[inds], features[inds], scores[inds], labels[inds])
            keep = box_ops.remove_small_boxes(boxes, min_size=0.01)
            boxes, features, scores, labels = (boxes[keep], features[keep], scores[keep], labels[keep])
            keep = box_ops.batched_nms(boxes, scores, labels, self.detector.roi_heads.nms_thresh)
            keep = keep[:self.detector.roi_heads.detections_per_img]
            boxes, features, scores, labels = (boxes[keep], features[keep], scores[keep], labels[keep])
            all_boxes.append(boxes)
            all_features.append(features)
            all_scores.append(scores)
            all_labels.append(labels)
        return (all_boxes, all_features, all_scores, all_labels)
