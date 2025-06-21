# generated_texts: [0.67, 0.9]
# answer: [0.5839080459770115, 0.847457627118644, 0.7724137931034483, 0.9449152542372882]

from main.eval_screenspot import get_bbox, pointinbbox

point = [0.79, 0.52]
bbox = [425, 280, 74, 20]
img_size = [640,447]


gt_bbox = get_bbox(bbox, img_size, False)

print(pointinbbox(point, gt_bbox))