from DGLD.common.dataset import GraphNodeAnomalyDectionDataset
from DGLD import get_parse
from DGLD.AAGNN import AAGNN_A
from DGLD.AAGNN import AAGNN_M
if __name__ == '__main__':
    args = get_parse.get_parse()
    print(args)
    dataset = GraphNodeAnomalyDectionDataset(args.dataset)
    model = AAGNN_A.model()
    model.fit(dataset[0], args)
    #pred_score = model.infer(dataset[1], args)
    #print(pred_score)
