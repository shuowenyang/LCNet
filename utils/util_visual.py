import dominate
from dominate.tags import *
import os

class HTML:
    def __init__(self, save_dir, file_name, reflesh=0):
        self.file_name = file_name
        self.save_dir = save_dir
        # self.img_dir = os.path.join(self.save_dir, 'images')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # if not os.path.exists(self.img_dir):
        #     os.makedirs(self.img_dir)

        self.doc = dominate.document(title=file_name)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def add_big_header(self, str):
        with self.doc:
            h1(str)

    def add_middle_header(self, str):
        with self.doc:
            h3(str)

    def add_small_header(self, str):
        with self.doc:
            h5(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, imgs, img_titles, links, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, title, link in zip(imgs, img_titles, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(title[:title.rfind(';')])
                            p(title[title.rfind(';') + 1:])

    def save(self):
        html_file = f'{self.save_dir}{self.file_name}'
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()