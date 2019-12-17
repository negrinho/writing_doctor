# TODO: it should be based on sentences.

import nltk
import json

# download resources needed.
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def split_sentence_into_word_tokens(sentence):
    return nltk.word_tokenize(sentence)


def tokenize_string_into_words(s):
    return nltk.word_tokenize(s)


def sort_dict_items(d, by_key=True, decreasing=False):
    key_fn = (lambda x: x[0]) if by_key else (lambda x: x[1])
    return sorted(d.items(), key=key_fn, reverse=decreasing)


def topk(key2cnt, k):
    return sort_dict_items(key2cnt, by_key=False, decreasing=True)[:k]


def read_textfile(filepath, strip=True):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if strip:
            lines = [line.strip() for line in lines]
        return lines


def write_textfile(filepath, lines, append=False, with_newline=True):
    mode = 'a' if append else 'w'

    with open(filepath, mode) as f:
        for line in lines:
            f.write(line)
            if with_newline:
                f.write("\n")


def read_jsonfile(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
        return d


def write_jsonfile(d, filepath, sort_keys=False, compactify=False):
    with open(filepath, 'w') as f:
        indent = None if compactify else 4
        json.dump(d, f, indent=indent, sort_keys=sort_keys)


def begin_end_indices_for_command_parens(s, cmd):
    # NOTE: might not work correctly if prefixes are shared by multiple commands
    idx = s.index("\\" + cmd)
    assert idx is not None
    depth = 0
    for i, ch in enumerate(s[idx:]):
        if ch == "{":
            if depth == 0:
                start_idx = idx + i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx + i
                return start_idx, end_idx


def string_inside_command(s, cmd):
    start_idx, end_idx = begin_end_indices_for_command_parens(s, cmd)
    return s[start_idx + 1:end_idx]


def strings_outside_command(s, cmd):
    start_idx, end_idx = begin_end_indices_for_command_parens(s, cmd)
    while s[start_idx] != "\\":
        start_idx -= 1
        if start_idx < 0:
            assert False
    return s[:start_idx], s[end_idx + 1:]


# TODO: something similar can be done for other envs, like paragraphs and sections.
# NOTE: these lines are untokenized still.
# NOTE: this is untested.
def parse_caption(lines, idx):
    assert lines[idx].startswith("\\caption{")
    depth = 0

    ### NOTE:
    out_lines = []
    while True:
        # print depth
        for i, ch in enumerate(lines[idx]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out_lines.append(lines[idx][:i + 1])
                    return out_lines
        out_lines.append(lines[idx])
        idx += 1


# TODO: make it easy to read captions..
# TODO: handle these better in these examples.
# TODO: tokenize inline math.
def read_latexfile(filepath,
                   ignore_comments=True,
                   lowercase=False,
                   has_begin_end_document=True,
                   tokenize_into_words=True,
                   tokenize_inline_math=True,
                   tokenize_commands_lst=None):
    lines = read_textfile(filepath, strip=True)
    if ignore_comments:
        lines = [s for s in lines if not s.startswith("%")]

    # remove environments
    if has_begin_end_document:
        start_idx = lines.index("\\begin{document}")
        end_idx = lines.index("\\end{document}")
        lines = lines[start_idx + 1:end_idx]

    out_lines = []
    inside_env_depth = 0
    # NOTE: hacky
    inside_itemize_depth = 0
    # inside_caption = False
    for i, s in enumerate(lines):
        if s.startswith("\\begin{itemize}"):
            inside_itemize_depth += 1
        elif s.startswith("\\end{itemize}"):
            inside_itemize_depth -= 1

        if s.startswith("\\begin{"):
            inside_env_depth += 1
        elif s.startswith("\\end{"):
            inside_env_depth -= 1
        elif s.startswith("\\caption"):
            out_lines.extend(parse_caption(lines, i))
        else:
            if inside_env_depth == 0 or inside_itemize_depth > 0:
                out_lines.append(s)
    lines = out_lines

    # join lines that are not separated by an empty line.
    out_lines = []
    line = []
    for s in lines:
        if s == '' and len(line) > 0:
            out_s = ' '.join(line)
            out_lines.append(out_s)
            line = []
        else:
            if s != '':
                line.append(s)
    lines = out_lines

    # tokenize math (TODO: make it work for display math; this does not check if $ are escaped or not.)
    if tokenize_inline_math:
        out_lines = []
        for s in lines:
            out_s = s
            while "$" in out_s:
                start_idx = out_s.index("$")
                end_idx = start_idx + out_s[start_idx + 1:].index("$")
                out_s = out_s[:start_idx] + " -MATH- " + out_s[end_idx + 2:]
            out_lines.append(out_s)
        lines = out_lines

    if tokenize_commands_lst is not None:
        out_lines = []
        for s in lines:
            out_s = s
            any_match = True
            while any_match:
                any_match = False
                for cmd in tokenize_commands_lst:
                    if "\\" + cmd in out_s:
                        # print out_s
                        x = out_s.index("\\" + cmd)

                        print x, out_s[x - 16:x + 16]
                        left_s, right_s = strings_outside_command(out_s, cmd)
                        out_s = left_s + " -" + cmd.upper() + "- " + right_s
                        any_match = True
            out_lines.append(out_s)
        lines = out_lines

    # tokenize into sentences and words
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    ps = [sent_detector.sentences_from_text(s) for s in lines]
    if tokenize_into_words:
        ps = [[nltk.word_tokenize(s) for s in p] for p in ps]
    if lowercase:
        ps = [[[w.lower() for w in s] for s in p] for p in ps]
    return ps


# TODO: add a file path
def read_latexfile_outline(filepath, include_paragraphs=True):
    lines = read_textfile(filepath, strip=True)
    out_lines = []
    for s in lines:
        match = False
        prefix = ""
        if s.startswith("\\section"):
            prefix = ""
            match = True
        elif s.startswith("\\subsection"):
            prefix = " " * 2
            match = True
        elif include_paragraphs and s.startswith("\\paragraph"):
            prefix = " " * 4
            match = True
        # NOTE: these are equivalent to paragraphs.
        elif include_paragraphs and s.startswith("\\item") and "\\textbf" in s:
            prefix = " " * 4
            match = True
        else:
            pass

        if match:
            s = prefix + s[s.index("{") + 1:s.index("}")]
            out_lines.append(s)
    return out_lines


# TODO: add an ignore comments option. for now, always ignores.
def read_markdown(filepath, lowercase=False, tokenize_into_words=True):
    lines = read_textfile(filepath, strip=True)

    # join lines that are not separated by an empty line.
    out_lines = []
    line = []
    is_inside_code = False
    for s in lines:
        if s == '' and len(line) > 0 and not is_inside_code:
            out_s = ' '.join(line)
            out_lines.append(out_s)
            line = []
        else:
            if s != '' and not is_inside_code:
                line.append(s)

        if s.startswith("```"):
            is_inside_code = not is_inside_code

    # tokenize into sentences and words
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    ps = [sent_detector.sentences_from_text(s) for s in out_lines]
    if tokenize_into_words:
        ps = [[nltk.word_tokenize(s) for s in p] for p in ps]
    if lowercase:
        ps = [[[w.lower() for w in s] for s in p] for p in ps]
    return ps


def get_todos(ps):
    ss = []
    for p in ps:
        for s in p:
            if "TODO" in s or "todo" in s:
                ss.append(s)
    return ss


def get_pos_tags_for_paragraphs(ps):
    return [[[x[1] for x in nltk.pos_tag(s)] for s in p] for p in ps]


def get_pos_tags_for_sentence(s):
    return [x[1] for x in nltk.pos_tag(s)]


def apply_to_paragraphs(ps, sentence_fn):
    return [[sentence_fn(s) for s in p] for p in ps]

def get_adjective_counts(ps):
    w2c = {}
    for p in ps:
        for s in p:
            tags = get_pos_tags_for_sentence(s)
            for i, t in enumerate(tags):
                if t.startswith("JJ"):
                    w = s[i]
                    if w not in w2c:
                        w2c[w] = 0
                    w2c[w] += 1
    return w2c

def get_adjective_counts(s):
    w2c = {}
    for p in ps:
        for s in p:
            tags = get_pos_tags_for_sentence(s)
            for i, t in enumerate(tags):
                if t.startswith("RB"):
                    w = s[i]
                    if w not in w2c:
                        w2c[w] = 0
                    w2c[w] += 1
    return w2c

def show_sentences(ps, str_fn, print_to_terminal=True):
    out_lines = []
    for i, p in enumerate(ps):
        for s in p:
            out = str_fn(s)
            if len(out) > 0:
                s = "%d => %s" % (i, out)
                out_lines.append(s)
                if print_to_terminal:
                    print s
    return out_lines


def bad_words_fn(s):
    lst = []
    for i in range(1, n_longest + 1)[::-1]:
        for j in xrange(0, len(s) - i + 1):
            x = tuple(s[j:j + i])
            if x in words:
                lst.append(x)
    if len(lst) > 0:
        return "%s: (%s)" % (str(lst), " ".join(s))
    else:
        return ""


def adjectives_fn(s):
    tags = get_pos_tags_for_sentence(s)
    lst = []
    for i, t in enumerate(tags):
        if t.startswith("JJ"):
            lst.append((s[i],))
    if len(lst) > 0:
        return "%s: (%s)" % (str(lst), " ".join(s))
    else:
        return ""


def adverbs_fn(s):
    tags = get_pos_tags_for_sentence(s)
    lst = []
    for i, t in enumerate(tags):
        if t.startswith("RB"):
            lst.append((s[i],))
    if len(lst) > 0:
        return "%s: (%s)" % (str(lst), " ".join(s))
    else:
        return ""


def repeats_fn(s):
    lst = []
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            lst.append(s[i])
    if len(lst) > 0:
        return "%s: (%s)" % (str(lst), " ".join(s))
    else:
        return ""


def get_frequent_long_words(ps, len_thres, count_thres, top_k):
    w2c = {}
    for p in ps:
        for s in p:
            for w in s:
                if len(w) >= len_thres:
                    if w not in w2c:
                        w2c[w] = 0
                    w2c[w] += 1
    w2c = {k: v for k, v in w2c.iteritems() if v >= count_thres}
    return topk(w2c, top_k)


# # NOTE: the API design needs to be revised.
# # TODO: add something that keep the paragraph structure and something that does not.
def sort_sentences_by_word_length(ps, filepath=None, print_to_terminal=True):
    lst = sorted([s for p in ps for s in p], key=lambda s: len(s), reverse=True)
    lines = ["  >>> " + " ".join(s) for s in lst]

    if filepath is not None:
        write_textfile(filepath, lines)
    if print_to_terminal:
        for s in lines:
            print s
    return lines


def sort_sentences_by_char_length(ps, filepath=None, print_to_terminal=True):
    lst = sorted([s for p in ps for s in p],
                 key=lambda s: sum([len(w) for w in s]) + len(s) - 1,
                 reverse=True)
    lines = ["  >>> " + " ".join(s) for s in lst]

    if filepath is not None:
        write_textfile(filepath, lines)
    if print_to_terminal:
        for s in lines:
            print s
    return lines


def show_paragraph_summary(ps, filepath=None, print_to_terminal=True):
    lines = []
    for p in ps:
        out_s = "|>>> " + " ".join(p[0])
        lines.append(out_s)
    # output to a file.
    if filepath is not None:
        write_textfile(filepath, lines)
    if print_to_terminal:
        for s in lines:
            print s
    return lines


# TODO: improve formatting overall.
### TODO: have a way of doing filepaths outputs. just add the arguments.
def show_paragraph_sentence_word_summary(ps,
                                         num_words,
                                         filepath=None,
                                         print_to_terminal=True):
    lines = []
    for p in ps:
        out_s = "|>>> " + " ".join(p[0])
        lines.append(out_s)
        for s in p[1:]:
            out_s = "  |- " + " ".join(s[:num_words])
            lines.append(out_s)
        lines.append("")
    # output to a file.
    if filepath is not None:
        write_textfile(filepath, lines)
    if print_to_terminal:
        for s in lines:
            print s
    return lines


def show_paragraph_sentence_char_summary(ps,
                                         num_chars,
                                         filepath=None,
                                         print_to_terminal=True):
    lines = []
    for p in ps:
        out_s = "|>>> " + " ".join(p[0])
        lines.append(out_s)
        for s in p[1:]:
            out_s = "  |- " + " ".join(s)[:num_chars]
            lines.append(out_s)
        lines.append("")
    # output to a file.
    if filepath is not None:
        write_textfile(filepath, lines)
    if print_to_terminal:
        for s in lines:
            print s
    return lines


def show_sentence_word_summary(ps,
                               num_words,
                               filepath=None,
                               print_to_terminal=True):
    lines = []
    for p in ps:
        for s in p:
            out_s = " ".join(s[:num_words])
            lines.append(out_s)
        lines.append("")
    # output to a file.
    if filepath is not None:
        write_textfile(filepath, lines)
    if print_to_terminal:
        for s in lines:
            print s
    return lines


def show_sentence_char_summary(ps,
                               num_chars,
                               filepath=None,
                               print_to_terminal=True):
    lines = []
    for p in ps:
        for s in p:
            out_s = " ".join(s)[:num_chars]
            lines.append(out_s)
        lines.append("")
    # output to a file.
    if filepath is not None:
        write_textfile(filepath, lines)
    if print_to_terminal:
        for s in lines:
            print s
    return lines


def show_sentences_with_words(ps, words, filepath=None, print_to_terminal=True):
    lines = []
    for p in ps:
        for s in p:
            for w in words:
                if w in s:
                    out_s = "  |- " + " ".join(s)
                    lines.append(out_s)
    if filepath is not None:
        write_textfile(filepath, lines)
    if print_to_terminal:
        for s in lines:
            print s
    return lines


ds = read_jsonfile("map.json")
words = set()
for d in ds:
    for s in d["map"]:
        k = tuple(tokenize_string_into_words(s))
        words.add(k)

n_longest = max([len(x) for x in words])

# ps = read_latexfile(
#     '/Users/negrinho/Desktop/5cdeeeb63c25b9478af185b2/deep_architect.tex',
#     lowercase=True, tokenize_inline_math=True,
#     tokenize_commands_lst=["ref", "cite"])

# TODO: allow this to deal wtih folders more neatly.

# "/Users/negrinho/Desktop/lourenco/Thesis_Introduction-pedro.tex"
# "/Users/negrinho/Desktop/lourenco/Thesis_Background_ObjectDetection.tex"
# filepath = "/Users/negrinho/Desktop/r2ai/_posts/2019-07-26-introducing-deep-architect.md"
# filepath = "/Users/negrinho/Desktop/r2ai/_posts/2019-12-01-on-writing.md"
# ps = read_markdown(filepath, lowercase=True)


# filepath = "/Users/negrinho/Desktop/neurips2019_deep_architect/deep_architect.tex"
# filepath = "/Users/negrinho/Desktop/writing_linting/object_detection.tex"
filepath = "/Users/negrinho/Desktop/beam_learn/papers/5cebae758c00615e225a1827/naaclhlt2019.tex"
ps = read_latexfile(filepath, lowercase=True, has_begin_end_document=True, ignore_comments=True)

out_lines = []
out_lines.extend(["**************** Section outline ****************", ""])
lst = read_latexfile_outline(filepath, include_paragraphs=True)
out_lines.extend(lst)
out_lines.append("")
out_lines.extend(
    ["**************** Suggested substitutions ****************", ""])
lst = show_sentences(ps, bad_words_fn, print_to_terminal=False)
out_lines.extend(lst)
out_lines.append("")
out_lines.extend(["**************** Paragraph summary ****************", ""])
lst = show_paragraph_summary(ps, print_to_terminal=False)
out_lines.extend(lst)
out_lines.append("")
# show_sentences(ps, adjectives_fn)
out_lines.extend(["**************** Summary ****************", ""])
lst = show_paragraph_sentence_word_summary(ps, 16, print_to_terminal=False)
out_lines.extend(lst)
out_lines.append("")
out_lines.extend(["**************** Adjectives ****************", ""])
lst = show_sentences(ps, adjectives_fn, print_to_terminal=False)
out_lines.extend(lst)
out_lines.append("")
out_lines.extend(["**************** Repeats ****************", ""])
lst =  show_sentences(ps, repeats_fn, print_to_terminal=False)
out_lines.extend(lst)
out_lines.append("")
write_textfile("summary.txt", out_lines)



# read_latexfile_outline(filepath)

#### TODO: make some of these easier to do.
# NOTE: there are some quadratic operations. I think that they have to share a lot of words at least.
# NOTE: usually a fairly small number of sentences.

# pprint(get_frequent_long_words(ps, 8, 0, 128))


# show_sentences_with_words(ps, ["module", "modules"], "module_summary.txt")
# show_sentences_with_words(ps, ["hyperparameter", "hyperparameters"], "hyperparameter_summary.txt")
# show_sentences_with_words(ps, ["assign", "assigns", "assignment"], "assign_summary.txt")
# show_sentences_with_words(ps, ["search"], "search_summary.txt")

# TODO: have a way of generating a simplified version.

# ps = read_latexfile(
#     '/Users/negrinho/Desktop/5cdeeeb63c25b9478af185b2/deep_architect.tex',
#     lowercase=True, tokenize_inline_math=True,
#     tokenize_commands_lst=["ref", "cite"])

# # TODO: fix the problem with tokenizing code. it does not seem to be working.

# sort_sentences_by_word_length(ps, "word_len.txt")
# sort_sentences_by_char_length(ps, "char_len.txt")

# show_sentences(ps, hyperp_fn)

# read_latexfile_outline('/Users/negrinho/Desktop/5cdeeeb63c25b9478af185b2/deep_architect.tex')
# read_latexfile_outline('/Users/negrinho/Desktop/5d24e732c489ea01dde4a1a7/main.tex', include_paragraphs=True)

# from pprint import pprint

# pprint([" ".join(s) for s in get_todos(read_la texfile(
#     '/Users/negrinho/Desktop/5cdeeeb63c25b9478af185b2/deep_architect.tex', ignore_comments=False))])

# TODO: mention that these things allow us to do multiple readings of the paper
# on different aspects.

# TODO: get text out of itemize. for those that involve itemize

# TODO: just remove stuff from it.

# TODO: handling the captions separately. for specific words.

# show_sentences()
# TODO: it is dealing directly with paragraphs for convenience. not ideal, but still...

# most frequent long words.

# paragraph with the most repetition of the

# get this information from some online datasets about synonyms and what not.

# TODO: to collect data, we can use something like a continuous
# running git that will capture differences between sentences.
# if we have a bunch of snapshots, I think that it
# should work.
# problem with in the middle of it.
# (fewer high quality snapshots vs many lower quality ones.)

# ----

# NOTE: probability of having me editing a paragraph.

# TODO: have a way of visualizing the content words.
# maybe through TF-IDF; maybe through something else (e.g., the most infrequent word in the sentence.)

# TODO: eventually do some annotation directly in a pdf and od the extraction directly from a pdf.

# TODO: show the first sentence of each paragraph.

# TODO: look at sentences with small edit distance. these might need to be normalized by the size of the sentence. max(l1, l2) (or something like this; also needs to deal with transpositions if possible, insertions and deletions might be fine by themselves.)

# TODO: maintaining this structure, it should be easy to keep this structure.

# TODO: clean up all the JSON files. just a basic read, and little else, don't even try to parse the strings for each field.

# NOTE: to merge the bib files and what not.

# TODO: come up with levels. better to do categories.
# warn, strict.

# python writing_linter.py --json_config xxx --levels='["a", "b", "c"]'

# TODO: change the way that it is shown there.

# TODO: introduce basic stuff with adjectives.
# TODO: makes more sense to write things in the file at the end.
# NOTE: makes it easier to use too.

# NOTE: some are tokenized, some are not.
# distinguish this. tk_ps, vs ps, lines, tk_lines, tk_out_lines.
# ps are always tokenized, I think.

# NOTE: I would say, some of them are responsible to return things in a format that is directly printable.

# TODO: for this, make it easy to generate these reduced documents.

# add words that end in ed,

# TODO: needs to be kept into different models. this is important.
#

# there are a few of these that truly require traversal to work.
# TODO: this is needed to get it to wokr

# words ending in ition are also interesting.

# TODO: have a way of extracting comments that match a certain thing.

# TODO: add the part of the model that repeats things.