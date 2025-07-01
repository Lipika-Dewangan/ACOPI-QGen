from models.bert_encoder import BertEncoder
from models.rgat import RGAT
from models.scil import SCILModule
from models.decoder import NonAutoDecoder
from utils.dependency_parser import parse_dependencies
from utils.saodt import build_saodt_tree
from utils.losses import compute_total_loss

def main():
    print("ACOPI-QGen Framework Initialized")
    # Load your config, data, model components and run training/inference here
    pass

if __name__ == "__main__":
    main()
