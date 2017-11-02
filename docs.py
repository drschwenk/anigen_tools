video_mkd_template = """## Video ID {}
![animation_frames]({})

![bounding_boxes]({})

![animation]({})

#### Interpolation
![interpolation]({})

#### Tracking
![tracking]({})

#### Segmentation
{} 

### Description
{}

#### Setting
{}

#### Characters
{}

#### Objects
{}

### Parse

#### noun phrase chunks
{}

#### coreference clusters
{}

- - -
"""

s3_doc_base_uri = 'https://s3-us-west-2.amazonaws.com/ai2-vision-animation-gan/documentation/images/'
local_path = '/Users/schwenk/wrk/animation_gan/ai2-vision-animation-gan/documentation/images/'
img_base_path = 'https://s3-us-west-2.amazonaws.com/ai2-vision-animation-gan/documentation/images/'


def paginate_docs(videos, page_size=50):
    num_sort = sorted(videos, key=lambda x: x.gid())
    for i in range(0, len(num_sort), page_size):
        yield num_sort[i:i + page_size]


def write_mkd_doc(doc, fp):
    with open(fp, 'w') as f:
        f.write(doc)


def format_characters(id_name_pairs):
    char_base = ''
    for char_id, char_name in id_name_pairs:
        char_base += '\tcharacter ' + char_id.rsplit('_', maxsplit=1)[-1] + ': ' + char_name + '\n\n'
    return char_base


def format_objects(id_name_pairs):
    char_base = ''
    for char_id, char_name in id_name_pairs:
        char_base += '\tcharacter ' + char_id.rsplit('_', maxsplit=1)[-1] + ': ' + char_name + '\n\n'
    return char_base


def gen_and_save_doc_images(video):
    try:
        three_frames = video.display_keyframes()
        frame_bboxes = video.display_bounding_boxes()

        three_frames.save(local_path + video.gid() + '_keyframes.png')
        frame_bboxes.save(local_path + video.gid() + '_bboxes.png')
    except:
        print(video.gid())


def draw_parse_trees(video):
    parsed_sents = parse_video(video, nlp, core_parser)
    for idx, sent in enumerate(parsed_sents):
        tree_name = local_path + video.gid() + '_sent_' + str(idx) + '_parse_tree'
        TreeView(sent)._cframe.print_to_file(tree_name + '.ps')
        _ = os.system('convert ' + tree_name + '.ps ' + tree_name + '.png')
        _ = os.system('rm ' + tree_name + '.ps')


def gen_video_mkd(video):
    segm_gif_str = ''
    for ent in video.data()['characters'] + video.data()['characters']:
        link = s3_doc_base_uri + ent.gid() + '_segm.gif'
        segm_gif_str += '![segmentation]({})\n\n'.format(link)
    entry_args = [
        video.gid(),
        s3_doc_base_uri + video.gid() + '_keyframes.png',
        s3_doc_base_uri + video.gid() + '_bboxes.png',
        video.display_gif(True),
        s3_doc_base_uri + video.gid() + '_interp.gif',
        s3_doc_base_uri + video.gid() + '_tracking.gif',
        segm_gif_str,
        video.description(),
        '\t' + video.setting(),
        format_characters(video.characters_present()),
        '\n\n'.join([': '.join(obj.data()['localID'].split('_', 1)) for obj in video.data()['objects']]),
        # '\n\n'.join(
        #     '![con_parse]({})'.format(s3_doc_base_uri + video.gid() + '_sent_' + str(sent_idx) + '_parse_tree.png') for
        #     sent_idx in range(len(video.data()['parse']['constituent_parse']))),
        '\t' + '\n\n\t'.join(video.data()['parse']['noun_phrase_chunks']['named_chunks']),
        '\t' + '\n\n\t'.join([' : '.join(cluster) for cluster in video.data()['parse']['coref']['named_clusters']])
    ]

    return video_mkd_template.format(*entry_args)


def doc_video_group(dataset, make_images=False):
    page_filenames = {}
    for idx, videos in enumerate(list(paginate_docs(dataset, 10))):
        idx += 1
        page_md = '\n\n\n\n'.join([gen_video_mkd(vid) for vid in videos])
        page_name = 'video group {0:0{width}}'.format(idx, width=2)
        page_filenames[page_name] = page_name.replace(' ', '_') + '.md'
        if make_images:
            _ = [gen_and_save_doc_images(vid) for vid in videos]
            # _ = [draw_parse_trees(vid) for vid in videos]
        write_mkd_doc(page_md, './documentation/docs/' + page_filenames[page_name])
    write_mkdocs_config(page_filenames, 'Flintstones Dataset Explorer')


def write_mkdocs_config(page_names, site_name):
    yml_head = """site_name: {}
theme: null
theme_dir: './material'
extra:
   font:
      text: Oxygen
      code: Oxygen
pages:
   - home: index.md
""".format(site_name)

    padding1 = '   '
    padding2 = '    '
    with open('./documentation/mkdocs.yml', 'w') as f:
        f.write(yml_head)
        for k, v in sorted(page_names.items()):
            f.write(''.join([padding1, '- ', k, ': ', v, '\n']))
            #         for k, vals in sorted(category_group_names.items()):
            #             f.write(''.join([padding1, '- ', k, ': ', '\n']))
            #             for v in vals:
            #                 f.write(''.join([padding2, '- ', v, ': ', v, '\n']))