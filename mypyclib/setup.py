from setuptools import setup

from mypyc.build import mypycify

setup(
    name='mypyclib',
    packages=['mypyclib'],
    ext_modules=mypycify(
        ['mypyclib/__init__.py',
        'mypyclib/fast_copy_batch_block_offsets.py',]
    ),
)
