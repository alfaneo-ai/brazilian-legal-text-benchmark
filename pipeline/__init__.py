import argparse

from .evaluation import EvaluationCommand


def parse_commands():
    parser = argparse.ArgumentParser(prog='Juridics', description='Similaridade de documentos juridicos')
    parser.add_argument('model', choices=['use', 'bm25', 'sbert'],
                        action='store', help='Modelos para calculo de similaridade')
    args = vars(parser.parse_args())
    return args


def run_command():
    args = parse_commands()
    model = args['model']
    command = EvaluationCommand(model)
    command.execute()
