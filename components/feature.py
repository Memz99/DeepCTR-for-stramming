from collections import OrderedDict, namedtuple, defaultdict

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'index', 'vocabulary_size', 'embedding_dim', 'dimension','sparse_embedding', 'dtype', 'embedding_name',
                             'group'])):
    __slots__ = ()

    def __new__(cls, name, index, vocabulary_size, embedding_dim=4, sparse_embedding=False, dtype="int32",
                embedding_name=None, group=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        return super(SparseFeat, cls).__new__(cls, name, index, vocabulary_size, embedding_dim, embedding_dim,
                                              sparse_embedding, dtype, embedding_name, group)


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'index', 'combiner', 'length_name'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, index, combiner="mean", length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, index, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def sparse_embedding(self):
        return self.sparsefeat.sparse_embedding

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name


class DenseFeat(namedtuple('DenseFeat', ['name', 'index', 'dimension', 'dtype', 'group'])):
    __slots__ = ()

    def __new__(cls, name, index, group, dtype="float32"):
        dimension = index[1] - index[0]
        return super(DenseFeat, cls).__new__(cls, name, index, dimension, dtype, group)

    def __hash__(self):
        return self.name.__hash__()


class Group(object):
    def __init__(self, feats):
        self.name_dict = defaultdict(list)
        for feat in feats:
            self.name_dict[feat.group].append(feat.name)
        self.feat_dict = defaultdict(list)
        for feat in feats:
            self.feat_dict[feat.group].append(feat)

    def get(self, *groups):
        return sum([self.name_dict[group] for group in groups], [])

    def get_name(self, *groups):
        return sum([self.name_dict[group] for group in groups], [])

    def get_fc(self, *groups):
        return sum([self.feat_dict[group] for group in groups], [])
