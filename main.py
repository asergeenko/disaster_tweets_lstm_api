import argparse

from core import config, logging, predict, get_model

parser = argparse.ArgumentParser(description="Disaster tweet classification")
for arg in ['message']:
    parser.add_argument("--"+arg, dest=arg)


def main(default_config):

    # Parse command line arguments
    args = parser.parse_args()

    # Load or create the model
    model = get_model()

    # Predict
    if args.message:
        print (args.message)
        print ('Message: %s\n Disaster tweet: %d'%(args.message, predict(args.message,model)))

if __name__ == "__main__":
    try:
        main(config)
    except BaseException as e:
        logging.exception(e)







