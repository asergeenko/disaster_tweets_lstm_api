import argparse

from core import config, logging, predict, get_model,train_pipeline

parser = argparse.ArgumentParser(description="Disaster tweet classification")
for arg in ['predict_message','train_file']:
    parser.add_argument("--"+arg, dest=arg)


def main(default_config):

    # Parse command line arguments
    args = parser.parse_args()

    if args.train_file:
        try:
            model = train_pipeline(args.train_file)
        except Exception as e:
            message = 'Error while training model'
            logging.exception(message)
            print(message+'\n'+str(e))
        else:
            print('Model has trained successfully')
    # Predict
    if args.predict_message:
        # Load or create the model
        model = get_model()
        print('Message: %s\n Disaster tweet: %d' % (args.predict_message, predict(args.predict_message, model)))




if __name__ == "__main__":
    try:
        main(config)
    except BaseException as e:
        logging.exception(e)







