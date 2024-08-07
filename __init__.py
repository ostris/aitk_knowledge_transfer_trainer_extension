# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


# This is for generic training (LoRA, Dreambooth, FineTuning)
class KnowledgeTransferExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "knowledge_transfer"

    # name is the name of the extension for printing
    name = "Knowledge Transfer Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .KnowledgeTransferTrainer import KnowledgeTransferTrainer
        return KnowledgeTransferTrainer

# This is for generic training (LoRA, Dreambooth, FineTuning)
class KnowledgeTransferPrunerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "knowledge_transfer_pruner"

    # name is the name of the extension for printing
    name = "Knowledge Transfer Trainer pruner"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .KnowledgeTransferPruner import KnowledgeTransferPruner
        return KnowledgeTransferPruner

# This is for generic training (LoRA, Dreambooth, FineTuning)
class KnowledgeTransferChAdderExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "knowledge_transfer_ch_adder"

    # name is the name of the extension for printing
    name = "Knowledge Transfer Trainer pruner"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .KnowledgeTransferChannelAdder import KnowledgeTransferChannelAdder
        return KnowledgeTransferChannelAdder


AI_TOOLKIT_EXTENSIONS = [
    # you can put a list of extensions here
    KnowledgeTransferExtension, KnowledgeTransferPrunerExtension, KnowledgeTransferChAdderExtension
]
