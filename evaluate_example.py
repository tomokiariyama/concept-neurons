from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type
import random
import logzero
from logzero import logger

# first initialize some hyperparameters
MODEL_NAME = "bert-base-uncased"

# to find the knowledge neurons, we need the same 'facts' expressed in multiple different ways, and a ground truth
TEXTS = [
    "[MASK] is the capital of Miyagi prefecture",
    "The most populous city in the Tohoku region is [MASK]",
    "Tohoku University is located in [MASK]",
    "[MASK] is famous for its beef tongue",
    "The Tohoku Rakuten Golden Eagles is a baseball team based in [MASK]",
    "In [MASK], the Pageant of Starlight is held in winter",
    "Yuzuru Hanyu is a figure skater from [MASK]",
]
TEXT = TEXTS[0]
GROUND_TRUTH = "Sendai"
"""
TEXTS = [
    "Sarah was visiting [MASK], the capital of france",
    "The capital of france is [MASK]",
    "[MASK] is the capital of france",
    "France's capital [MASK] is a hotspot for romantic vacations",
    "The eiffel tower is situated in [MASK]",
    "[MASK] is the most populous city in france",
    "[MASK], france's capital, is one of the most popular tourist destinations in the world",
]
TEXT = TEXTS[0]
GROUND_TRUTH = "paris"
"""

# these are some hyperparameters for the integrated gradients step
BATCH_SIZE = 20
STEPS = 20 # number of steps in the integrated grad calculation
ADAPTIVE_THRESHOLD = 0.2 # in the paper, they find the threshold value `t` by multiplying the max attribution score by some float - this is that float.
P = 0.5 # the threshold for the sharing percentage

# setup model & tokenizer
model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

# initialize the knowledge neuron wrapper with your model, tokenizer and a string expressing the type of your model ('gpt2' / 'gpt_neo' / 'bert')
kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))

# use the integrated gradients technique to find some refined neurons for your set of prompts
refined_neurons = kn.get_refined_neurons(
    TEXTS,
    GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
)

logzero.logfile('log/test_run.log')
logger.info(f'refining done')

# suppress the activations at the refined neurons + test the effect on a relevant prompt
# 'results_dict' is a dictionary containing the probability of the ground truth being generated before + after modification, as well as other info
# 'unpatch_fn' is a function you can use to undo the activation suppression in the model.
# By default, the suppression is removed at the end of any function that applies a patch, but you can set 'undo_modification=False',
# run your own experiments with the activations / weights still modified, then run 'unpatch_fn' to undo the modifications
logger.info(f'suppress the activations at the refined neurons + test the effect on a relevant prompt')
results_dict, unpatch_fn = kn.suppress_knowledge(
    TEXT, GROUND_TRUTH, refined_neurons
)


# suppress the activations at the refined neurons + test the effect on an unrelated prompt
logger.info(f'suppress the activations at the refined neurons + test the effect on an unrelated prompt')
results_dict, unpatch_fn = kn.suppress_knowledge(
    "[MASK] is the official language of the solomon islands",
    "english",
    refined_neurons,
)


# enhance the activations at the refined neurons + test the effect on a relevant prompt
logger.info(f'enhance the activations at the refined neurons + test the effect on a relevant prompt')
results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)


# enhance the activations at the refined neurons + test the effect on a unrelated prompt
logger.info('enhance the activations at the refined neurons + test the effect on an unrelated prompt')
results_dict, unpatch_fn = kn.enhance_knowledge(
    "[MASK] is the official language of the solomon islands",
    "english",
    refined_neurons,
)


# erase the weights of the output ff layer at the refined neurons (replacing them with zeros) + test the effect
logger.info(f'erase the weights of the output ff layer at the refined neurons (replacing them with zeros) + test the effect')
results_dict, unpatch_fn = kn.erase_knowledge(
    TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="zero"
)


# erase the weights of the output ff layer at the refined neurons (replacing them with an unk token) + test the effect
logger.info(f'erase the weights of the output ff layer at the refined neurons (replacing them with an unk token) + test the effect')
results_dict, unpatch_fn = kn.erase_knowledge(
    TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="unk"
)


# edit the weights of the output ff layer at the refined neurons (replacing them with the word embedding of 'target') + test the effect
# we can make the model think the capital of france is London!
logger.info(f"edit the weights of the output ff layer at the refined neurons (replacing them with the word embedding of 'target') + test the effect")
results_dict, unpatch_fn = kn.edit_knowledge(
    TEXT, target="london", neurons=refined_neurons
)
