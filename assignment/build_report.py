# ADDED: new file (FMGAI assignment) - renders the Phase 2 report PDF.
"""Build assignment_results/phase2/Phase2_Report.pdf.

Every number in the report is read from the JSON/CSV files written by the runs
and by assignment/build_phase2_artifacts.py, so run that script first.
"""
import csv
import difflib
import json
import os
import subprocess
from itertools import zip_longest

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = 'assignment_results'
P2 = os.path.join(ROOT, 'phase2')
FIG = os.path.join(P2, 'figures')
COMMIT = '01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d'
UPSTREAM = 'https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement'
BLOB = UPSTREAM + '/blob/' + COMMIT
FORK = 'https://github.com/rikin-patel/sr3_diffusion'
WANDB = 'https://wandb.ai/rikin-patel-indian-institute-of-technology-jodhpur/FMGAI-Assn'

GROUP = '[GROUP NUMBER / NAME - to be filled in]'
MEMBERS = [('Rikin Patel', 'M26AI1096'), ('Anshul', 'M26AI1020'),
           ('Sonalika Sharma', 'M26AI1120'), ('Balasubramanium M', 'M26AI1030')]
RESPONSIBILITY = '[COMPONENT(S) - to be filled in]'

font_dir = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('Body', os.path.join(font_dir, 'DejaVuSerif.ttf')))
pdfmetrics.registerFont(TTFont('Body-Bold', os.path.join(font_dir, 'DejaVuSerif-Bold.ttf')))
pdfmetrics.registerFont(TTFont('Body-Italic', os.path.join(font_dir, 'DejaVuSerif-Italic.ttf')))
pdfmetrics.registerFont(TTFont('Mono', os.path.join(font_dir, 'DejaVuSansMono.ttf')))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic')

BODY = ParagraphStyle('body', fontName='Body', fontSize=9.3, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=5)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=8.2, leading=10.8, alignment=0, spaceAfter=0)
CELLC = ParagraphStyle('cellc', parent=CELL, alignment=TA_CENTER)
EQ = ParagraphStyle('eq', parent=BODY, alignment=TA_CENTER, spaceBefore=1, spaceAfter=6)
H1 = ParagraphStyle('h1', fontName='Body-Bold', fontSize=15, leading=19, spaceAfter=3)
H2 = ParagraphStyle('h2', fontName='Body-Bold', fontSize=11.4, leading=15, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor('#1f4e79'))
H3 = ParagraphStyle('h3', fontName='Body-Bold', fontSize=9.6, leading=13, spaceBefore=5, spaceAfter=3)
CAP = ParagraphStyle('cap', parent=BODY, fontName='Body-Italic', fontSize=8.2, leading=11, alignment=TA_CENTER, spaceAfter=8)
LI = ParagraphStyle('li', parent=BODY, leftIndent=13, firstLineIndent=-13, spaceAfter=2.5)
CODE = ParagraphStyle('code', fontName='Mono', fontSize=5.5, leading=6.9, wordWrap='CJK')
CODEH = ParagraphStyle('codeh', parent=CELL, fontName='Body-Bold', fontSize=7.2, leading=9)

W = A4[0] - 4.0 * cm
AB = 'ᾱ<sub>t</sub>'
TH = 'θ<sub>t</sub>'


def P(text, style=BODY):
    return Paragraph(text, style)


def link(url, text=None):
    return '<link href="{}" color="#1f4e79"><u>{}</u></link>'.format(url, text or url)


def code(text):
    return '<font name="Mono" size="7.8">{}</font>'.format(text)


def table(rows, widths, header=True, center_from=None, style=CELL, bold_rows=()):
    data = []
    for ri, r in enumerate(rows):
        line = []
        for ci, c in enumerate(r):
            st = CELLC if (center_from is not None and ci >= center_from) else style
            txt = str(c)
            if ri in bold_rows:
                txt = '<b>{}</b>'.format(txt)
            line.append(Paragraph(txt, st))
        data.append(line)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9aa5b1')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
          ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if header:
        st.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef4')))
    t.setStyle(TableStyle(st))
    return t


def figure(name, caption, width=W):
    path = os.path.join(FIG, name)
    iw, ih = ImageReader(path).getSize()
    return KeepTogether([Image(path, width=width, height=width * ih / iw), Spacer(1, 2), P(caption, CAP)])


def sci(v, digits=3):
    mant, exp = '{:.{}e}'.format(v, digits).split('e')
    return '{} × 10<super>{}</super>'.format(mant, int(exp))


def pm(mean, std, nd):
    return '{:.{n}f} ± {:.{n}f}'.format(mean, std, n=nd)


def load_json(*parts):
    with open(os.path.join(*parts)) as f:
        return json.load(f)


# ------------------------------------------------------------------ code diff
MODIFIED_FILES = ['sr.py', 'model/sr3_modules/diffusion.py', 'model/model.py',
                  'core/logger.py', 'core/wandb_logger.py', 'data/__init__.py']


def file_hunks(path):
    old = subprocess.check_output(['git', 'show', '{}:{}'.format(COMMIT, path)]).decode('utf-8').splitlines()
    with open(path, encoding='utf-8') as f:
        new = f.read().splitlines()
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return old, new, [op[1:] for op in sm.get_opcodes() if op[0] != 'equal']


def ranges(pairs):
    return ', '.join(str(a) if a == b else '{}-{}'.format(a, b) for a, b in pairs)


def code_cell(lineno, text):
    if text is None:
        return Paragraph('', CODE)
    esc = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    lead = len(esc) - len(esc.lstrip(' '))
    esc = '&nbsp;' * lead + esc.lstrip(' ')
    return Paragraph('<font color="#7a7a7a">{:>3}</font>&nbsp;{}'.format(lineno, esc), CODE)


def side_by_side(path):
    old, new, hunks = file_hunks(path)
    data = [[Paragraph('ORIGINAL &nbsp; {} &nbsp; (commit {})'.format(path, COMMIT[:7]), CODEH),
             Paragraph('MODIFIED &nbsp; {} &nbsp; (submitted version)'.format(path), CODEH)]]
    style = [('GRID', (0, 0), (-1, 0), 0.4, colors.HexColor('#9aa5b1')),
             ('BOX', (0, 0), (-1, -1), 0.4, colors.HexColor('#9aa5b1')),
             ('LINEAFTER', (0, 0), (0, -1), 0.4, colors.HexColor('#9aa5b1')),
             ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef4')),
             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
             ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3),
             ('TOPPADDING', (0, 0), (-1, -1), 0.6), ('BOTTOMPADDING', (0, 0), (-1, -1), 0.6)]
    for i1, i2, j1, j2 in hunks:
        span = lambda a, b: 'line {}'.format(a) if a == b else 'lines {}-{}'.format(a, b)
        left_label = span(i1 + 1, i2) if i2 > i1 else 'nothing here (new lines inserted after line {})'.format(i1)
        right_label = span(j1 + 1, j2) if j2 > j1 else 'removed'
        row = len(data)
        data.append([Paragraph(left_label, CODEH), Paragraph(right_label, CODEH)])
        style += [('BACKGROUND', (0, row), (-1, row), colors.HexColor('#f4f6f8')),
                  ('LINEABOVE', (0, row), (-1, row), 0.4, colors.HexColor('#9aa5b1'))]
        start = len(data)
        for k, (o, n) in enumerate(zip_longest(old[i1:i2], new[j1:j2])):
            data.append([code_cell(i1 + 1 + k, o), code_cell(j1 + 1 + k, n)])
        end = len(data) - 1
        if i2 > i1:
            style.append(('BACKGROUND', (0, start), (0, start + (i2 - i1) - 1), colors.HexColor('#fdecea')))
        if j2 > j1:
            style.append(('BACKGROUND', (1, start), (1, start + (j2 - j1) - 1), colors.HexColor('#eaf6ec')))
        del end
    t = Table(data, colWidths=[W * 0.46, W * 0.54], repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def main():
    num = load_json(P2, 'tables', 'phase2_numbers.json')
    env = load_json(P2, 'tables', 'environment.json')
    ds = load_json(P2, 'tables', 'dataset_stats.json')
    dec = load_json(P2, 'tables', 'error_decomposition.json')
    paired = load_json(ROOT, 'comparison', 'paired_differences.json')
    sch, ver, loss = num['schedule'], num['verification'], num['loss']
    val = {r['epoch']: r for r in num['validation_by_epoch']}
    summ = {k: load_json(ROOT, d, 'metrics_summary.json') for k, d in
            (('Run 0', 'run_0_baseline'), ('Run A', 'run_a_original_schedule'), ('Run B', 'run_b_power_schedule'))}
    x = '{:.2f}'.format(sch['x'])
    T = sch['T']
    Tn = '{:,}'.format(T)
    ab = paired['Run A -> Run B']
    a0 = paired['Run 0 -> Run A']
    b0 = paired['Run 0 -> Run B']
    below = sch['timesteps_of_power_schedule_below_native_terminal_level']
    t_cross = sch['first_t_where_power_schedule_is_below_native_terminal_level']
    assert sch['all_in_open_unit_interval']
    assert ver['Run B']['stored_buffer_matches_native_pow_x'] and ver['Run A']['stored_buffer_matches_native']

    doc = SimpleDocTemplate(os.path.join(P2, 'Phase2_Report.pdf'), pagesize=A4, leftMargin=2.0 * cm,
                            rightMargin=2.0 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                            title='Phase 2 Report - Noise schedule study on SR3', author=', '.join(m[0] for m in MEMBERS))
    e = []

    # ============================================================ title
    e.append(P('Fine-tuning SR3 on endoscopy images under a power-transformed noise schedule', H1))
    e.append(P('Phase 2 report, Foundation Models and Generative AI'))
    e.append(table([
        ['Group', GROUP],
        ['Members', '; '.join('{} ({})'.format(*m) for m in MEMBERS)],
        ['Code', link(FORK) + ', branch ' + code('feature/phase2-report')],
        ['Experiment logs', link(WANDB) + ' (Weights &amp; Biases project FMGAI-Assn)'],
    ], [3.0 * cm, W - 3.0 * cm], header=False))
    e.append(Spacer(1, 5))
    e.append(P(
        '<b>Summary.</b> We take the public SR3 checkpoint for 16×16 → 128×128 super-resolution, trained on face '
        'photographs, and apply it to Kvasir-SEG colonoscopy images. Without fine-tuning (Run 0) it reaches '
        '{:.2f} dB PSNR on our 100 test images. Fine-tuning for 100 epochs with the native schedule (Run A) gives '
        '{:.2f} dB, and fine-tuning with the schedule raised to the power x = {} (Run B) gives {:.2f} dB. Both '
        'fine-tuned runs are clearly better than Run 0. Run A is ahead of Run B by {:.2f} dB (paired standard error '
        '{:.2f}), while the perceptual score LPIPS is the same for both ({:.3f} and {:.3f}). Each run was trained '
        'once, with one seed.'.format(summ['Run 0']['psnr_mean'], summ['Run A']['psnr_mean'], x,
                                      summ['Run B']['psnr_mean'], -ab['psnr']['mean_difference'],
                                      ab['psnr']['standard_error'], summ['Run A']['lpips_mean'],
                                      summ['Run B']['lpips_mean'])))

    # ============================================================ 1
    e.append(P('1. Task and Model', H2))
    e.append(P(
        '<b>Task.</b> 8× single-image super-resolution of gastrointestinal endoscopy images, from 16×16 to 128×128 '
        'pixels. The input is a 16×16 RGB image made by bicubic downsampling of a colonoscopy frame. The output is a '
        '128×128 RGB image, and the target is the original frame resized and centre-cropped to 128×128.'))
    e.append(P(
        '<b>Model.</b> SR3 (Saharia et al., "Image Super-Resolution via Iterative Refinement", '
        + link('https://arxiv.org/abs/2104.07636', 'arXiv:2104.07636') + ') is a conditional denoising diffusion '
        'model. The authors did not release weights, so we use the open PyTorch implementation by Janspiry ('
        + link(UPSTREAM) + ', pinned at commit ' + code(COMMIT[:7]) + ') and its released 16×16 → 128×128 checkpoint '
        + code('I640000_E37') + ' (' + link(BLOB + '/README.md#L84', 'README.md, line 84') + '). The network is a U-Net '
        'with 97.8 million parameters. It receives the low-resolution image, upsampled to 128×128 with bicubic '
        'interpolation, concatenated channel-wise with the current noisy image.'))
    e.append(P(
        '<b>Forward process.</b> Let x<sub>0</sub> be the clean 128×128 image, c the bicubic-upsampled conditioning '
        'image, ε ~ N(0, I) Gaussian noise, and t = 1, …, T the timestep with T = ' + Tn + '. The per-step noise '
        'variances are β<sub>t</sub>, increasing linearly from 10<super>−6</super> to 10<super>−2</super>. With '
        'α<sub>t</sub> = 1 − β<sub>t</sub>, the forward process adds noise step by step, '
        'q(x<sub>t</sub> | x<sub>t−1</sub>) = N(√α<sub>t</sub> · x<sub>t−1</sub>, β<sub>t</sub> I), which gives the '
        'closed form'))
    e.append(P('x<sub>t</sub> = √' + AB + ' · x<sub>0</sub> + √(1 − ' + AB + ') · ε, &nbsp;&nbsp; with &nbsp;&nbsp;'
               + AB + ' = ∏<sub>s=1..t</sub> α<sub>s</sub> = ∏<sub>s=1..t</sub> (1 − β<sub>s</sub>).', EQ))
    e.append(P(
        'The network ε<sub>φ</sub>(c, x<sub>t</sub>, √' + AB + ') predicts the noise ε and is trained with an L1 loss, '
        '‖ε − ε<sub>φ</sub>(c, x<sub>t</sub>, √' + AB + ')‖<sub>1</sub>. Sampling starts from x<sub>T</sub> ~ N(0, I) and '
        'applies the learned reverse step T times.'))
    e.append(P(
        '<b>Native noise parameterization.</b> We take <b>' + TH + ' = ' + AB + '</b>, the cumulative product of the '
        'per-step signal-retention factors. The SR3 paper calls it γ<sub>t</sub> ("γ<sub>t</sub> = ∏<sub>i=1..t</sub> '
        'α<sub>i</sub>", Section 2.1, page 3) and states "Our model conditions on γ directly (vs t as in [17])" '
        '(Section 2.4, page 4). The code does the same: the forward sample is built from ' + AB + ' alone and the '
        'network receives √' + AB + ' instead of the timestep index ('
        + link(BLOB + '/model/sr3_modules/diffusion.py#L95-L106', 'diffusion.py') + ', lines 95-106, 153-157, 212-231 '
        'at the pinned commit). The config specifies the schedule through β<sub>t</sub>, and ' + AB + ' follows from '
        'it deterministically; we chose ' + AB + ' because it is the quantity both the forward process and the network '
        'actually consume.'))

    # ============================================================ 2
    e.append(P('2. Dataset and Verification', H2))
    e.append(P(
        '<b>Dataset.</b> Kvasir-SEG (Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MultiMedia Modeling 2020, '
        + link('https://arxiv.org/abs/1911.07069', 'arXiv:1911.07069') + '), downloaded from '
        + link('https://datasets.simula.no/kvasir-seg/') + '. It contains <b>{:,} colonoscopy images</b> of polyps, '
        'each with a segmentation mask. All {:,} files are RGB JPEGs. They come in {} different resolutions, with '
        'widths from {} to {} pixels and heights from {} to {} pixels; the smallest image is {}×{} and the largest '
        '{}×{}. We use the images only. The masks are not needed for super-resolution.'.format(
            ds['n_images'], ds['n_images'], ds['distinct_resolutions'], ds['width_min_max'][0], ds['width_min_max'][1],
            ds['height_min_max'][0], ds['height_min_max'][1], ds['smallest_image_wxh'][0], ds['smallest_image_wxh'][1],
            ds['largest_image_wxh'][0], ds['largest_image_wxh'][1])))
    e.append(P(
        '<b>Why it suits the task.</b> Every image is far larger than 128×128, so a true high-resolution target exists '
        'for each low-resolution input and nothing has to be upsampled to create the ground truth. The content (mucosa, '
        'folds, polyps, specular highlights) is rich in fine texture, which is exactly what 8× super-resolution has to '
        'reconstruct. And the domain is far from the faces the model was pretrained on, so the effect of fine-tuning '
        'is easy to isolate. With 1,000 images the dataset is also small enough to fine-tune on one laptop GPU.'))
    e.append(P(
        '<b>Evidence that it was not used in pretraining.</b> As in the declaration, we follow option (b). Because the '
        'weights come from the open implementation, its repository is the authoritative record of what this checkpoint '
        'was trained on. Line numbers refer to commit ' + code(COMMIT[:7]) + '.'))
    e.append(table([
        ['Source', 'Location', 'What it says'],
        ['Codebase, ' + link(BLOB + '/README.md#L24', 'README.md'), 'line 24',
         'Conditional model listed as "16×16 -> 128×128 on FFHQ-CelebaHQ".'],
        ['Codebase, ' + link(BLOB + '/README.md#L84', 'README.md'), 'line 84',
         'The released checkpoint we use is labelled "16×16 -> 128×128 on FFHQ-CelebaHQ".'],
        ['Codebase, ' + link(BLOB + '/config/sr_sr3_16_128.json#L16-L19', 'config/sr_sr3_16_128.json'), 'lines 16-19',
         'Training set: ' + code('"name": "FFHQ"') + ', ' + code('"dataroot": "dataset/ffhq_16_128"') + '.'],
        ['Codebase, ' + link(BLOB + '/config/sr_sr3_16_128.json#L28-L31', 'config/sr_sr3_16_128.json'), 'lines 28-31',
         'Validation set: ' + code('"name": "CelebaHQ"') + ', ' + code('"dataroot": "dataset/celebahq_16_128"') + '.'],
        ['Paper, ' + link('https://arxiv.org/abs/2104.07636', 'arXiv:2104.07636'),
         'page 5, Section 4, right column, lines 11-13 (paragraph beginning "Datasets:")',
         '"training face super-resolution models on Flickr-Faces-HQ (FFHQ) and evaluating on CelebA-HQ".'],
    ], [4.3 * cm, 3.9 * cm, W - 8.2 * cm]))
    e.append(Spacer(1, 4))
    e.append(P(
        'FFHQ and CelebA-HQ are collections of aligned, centred photographs of human faces taken with ordinary cameras '
        'in visible light. Kvasir-SEG images are frames recorded inside the large bowel with a medical endoscope. They '
        'show mucosa, folds, polyps, fluid and strong specular highlights from the endoscope light, usually inside a '
        'dark border and sometimes with an overlaid position marker. There are no faces, people or everyday objects in '
        'them, and their colours (mostly reds and pinks), textures and scene geometry differ from portrait photography. '
        'The pretraining pipeline reads only the two face folders named in the config, so Kvasir-SEG could not have '
        'entered pretraining.'))

    # ============================================================ 3
    e.append(P('3. Schedule Analysis', H2))
    e.append(P((
        '<b>' + TH + ' ∈ (0, 1) for all t.</b> Every β<sub>s</sub> lies in [10<super>−6</super>, 10<super>−2</super>], '
        'so every factor 1 − β<sub>s</sub> lies strictly between 0 and 1, and a finite product of such factors is again '
        'strictly between 0 and 1. Numerically, in double precision, ' + AB + ' decreases monotonically from '
        '<b>{:.6f}</b> at t = 1 to <b>{}</b> at t = {}. For x = {}, ' + AB + '<super>x</super> decreases from '
        '<b>{:.7f}</b> to <b>{}</b>, which is still strictly positive and representable in single precision. '
        'The training code asserts both bounds every time a schedule is built, and the ' + code('alphas_cumprod')
        + ' buffer stored in the pretrained checkpoint equals this native schedule.').format(
            sch['theta_range_native'][0], sci(sch['theta_range_native'][1]), Tn, x,
            sch['theta_range_power'][0], sci(sch['theta_range_power'][1]))))
    e.append(figure('schedule.png',
                    'Figure 1. Left: the native schedule θ<sub>t</sub> and the transformed schedule '
                    'θ<sub>t</sub><super>x</super> against t on the same axes. Right: the signal-to-noise ratio of '
                    'both schedules on a log scale.'))
    e.append(P('<b>What changes in the forward process when x &gt; 1.</b> Replacing ' + AB + ' by ' + AB
               + '<super>x</super> gives'))
    e.append(P('x<sub>t</sub> = √(' + AB + '<super>x</super>) · x<sub>0</sub> + √(1 − ' + AB + '<super>x</super>) · ε.', EQ))
    for item in [
        'Since 0 &lt; ' + AB + ' &lt; 1 and x &gt; 1, ' + AB + '<super>x</super> &lt; ' + AB + ' at every t. The signal '
        'is scaled down by a further factor ' + AB + '<super>(x−1)/2</super> and the noise variance rises from 1 − '
        + AB + ' to 1 − ' + AB + '<super>x</super>. The image is destroyed faster.',
        'The transformed process is still a valid Markov diffusion. Because (∏α<sub>s</sub>)<super>x</super> = '
        '∏α<sub>s</sub><super>x</super>, it has per-step factors α′<sub>t</sub> = α<sub>t</sub><super>x</super> and '
        'variances β′<sub>t</sub> = 1 − (1 − β<sub>t</sub>)<super>x</super> ≈ x · β<sub>t</sub>. Each step injects '
        'about x times as much noise; the largest step variance grows from {:.3f} to {:.4f}.'.format(
            sch['beta_max_native'], sch['beta_max_power']),
        'For this linear β schedule, −log ' + AB + ' grows almost exactly like t<super>2</super>, so ' + AB
        + '<super>x</super> = ᾱ<sub>τ</sub> with τ ≈ √x · t = {:.2f} t. The modified process at step t is at the noise '
        'level the native process only reaches at step {:.2f} t: step 200 corresponds to native step {}, step 400 to '
        '{}, and step 800 to {}.'.format(sch['x'] ** 0.5, sch['x'] ** 0.5, sch['equivalent_native_timestep_tau']['200'],
                                         sch['equivalent_native_timestep_tau']['400'],
                                         sch['equivalent_native_timestep_tau']['800']),
        ('As a result the modified process passes the native terminal noise level (' + AB + ' = {}) already at '
         't = {}. The remaining <b>{} of {} timesteps ({:.1f}%)</b> lie at noise levels below anything the pretrained '
         'network was trained on; its noise-level input √' + AB + ' was never smaller than {:.4f} in pretraining and '
         'goes down to {} in Run B.').format(sci(sch['theta_range_native'][1]), t_cross, below, Tn,
                                             100 * sch['fraction_below_native_terminal_level'],
                                             sch['sqrt_theta_min_native'], sci(sch['sqrt_theta_min_power'], 2)),
    ]:
        e.append(P('• ' + item, LI))
    e.append(P(
        '<b>Signal-to-noise ratio.</b> With SNR(t) = ' + AB + ' / (1 − ' + AB + '), the exponent changes it to '
        'SNR<sub>x</sub>(t) = ' + AB + '<super>x</super> / (1 − ' + AB + '<super>x</super>). The ratio of the two is'))
    e.append(P('SNR<sub>x</sub>(t) / SNR(t) = ' + AB + '<super>x−1</super> · (1 − ' + AB + ') / (1 − ' + AB
               + '<super>x</super>).', EQ))
    e.append(P((
        'Near t = 1, where ' + AB + ' → 1, we have 1 − ' + AB + '<super>x</super> ≈ x (1 − ' + AB + '), so the ratio '
        'tends to 1/x = {:.2f}: even the cleanest steps lose a factor x in SNR. As t grows the ratio falls towards '
        + AB + '<super>x−1</super>, which goes to zero. The SNR is therefore lowered at every timestep, by a factor '
        'that starts at 1/x and keeps shrinking. The point where signal and noise are equal (SNR = 1) moves from '
        't = {} to t = {}, so the number of timesteps with more signal than noise drops from {} to {}.').format(
            1 / sch['x'], sch['t_where_snr_drops_below_1']['native'], sch['t_where_snr_drops_below_1']['power'],
            sch['timesteps_with_snr_above_1']['native'], sch['timesteps_with_snr_above_1']['power'])))
    rows = [['t', AB + ' (native)', AB + '<super>' + x + '</super>', 'SNR (native)', 'SNR (x = ' + x + ')', 'ratio']]
    for k in ('1', '250', '500', '1000', '1500', '2000'):
        num_fmt = lambda v: '{:.7f}'.format(v) if v >= 0.01 else sci(v)
        rows.append(['{:,}'.format(int(k)), num_fmt(sch['theta_at'][k]['native']), num_fmt(sch['theta_at'][k]['power']),
                     sci(sch['snr_at'][k]['native']), sci(sch['snr_at'][k]['power']),
                     sci(sch['snr_ratio_power_over_native_at'][k], 2)])
    e.append(table(rows, [1.4 * cm] + [(W - 1.4 * cm) / 5] * 5, center_from=0))
    e.append(P('Table 1. Noise parameter and signal-to-noise ratio of both schedules at selected timesteps.', CAP))

    # ============================================================ 4
    e.append(P('4. Experimental Setup', H2))
    e.append(P(
        '<b>Dataset preparation.</b> We downloaded ' + code('kvasir-seg.zip') + ' from the Simula page and used the '
        + code('images/') + ' folder as it is. Cleaning consisted of checks rather than removals: all {:,} files open '
        'as RGB images, none is smaller than 128 pixels on its shorter side, and a SHA-256 comparison found no exact '
        'duplicate files, so no image was dropped. Each image is then processed the way the SR3 codebase prepares its '
        'own data: the shorter side is resized with bicubic interpolation and the image is centre-cropped to a square. '
        'This is done three times per image, giving the 128×128 target, the 16×16 input, and the 16×16 input '
        'upsampled back to 128×128 (the conditioning image). Pixel values are scaled to [−1, 1]. The only augmentation '
        'is a random horizontal flip during training, the codebase default.'.format(ds['n_images'])))
    assert ds['split_union_is_all_files'] and ds['split_overlap'] == 0 and ds['exact_duplicate_file_groups'] == 0
    e.append(P((
        '<b>Split.</b> {} training, {} validation and {} test images. The file names are sorted, shuffled once with '
        'Python\'s ' + code('random.Random(42)') + ', and cut at 800 and 900. The three file lists are stored in '
        + code('kvasir_data/splits/') + '. The splits are disjoint and together cover all {:,} images.').format(
            ds['split_sizes']['train'], ds['split_sizes']['val'], ds['split_sizes']['test'], ds['n_images'])))
    e.append(P('<b>Fine-tuning hyperparameters,</b> identical in Run A and Run B:'))
    e.append(table([
        ['Setting', 'Value'],
        ['Starting weights', 'Generator weights of ' + code('I640000_E37') + ' for both runs. The Adam state saved with '
                             'the checkpoint is not loaded; both runs start a new optimizer at iteration 0.'],
        ['Optimizer', 'Adam (PyTorch defaults, β = (0.9, 0.999)), constant learning rate 1 × 10<super>−4</super>, no weight decay'],
        ['Batch size', '4'],
        ['Length', '100 epochs = 20,000 iterations (200 iterations per epoch over 800 images)'],
        ['Loss', 'L1 between true and predicted noise'],
        ['Network', 'U-Net exactly as in the pretrained config, dropout 0.2, all weights trainable'],
        ['Diffusion steps', 'T = ' + Tn + ' for training and for sampling'],
        ['Random seed', '42 for Python, NumPy and PyTorch. With the same seed both runs see the same batches, the same '
                        'timestep draws and the same noise. cuDNN auto-tuning is left on, so results are not bit-reproducible.'],
        ['Only difference', 'Run A uses ' + AB + '; Run B uses ' + AB + '<super>' + x + '</super>, for training and for sampling.'],
    ], [3.3 * cm, W - 3.3 * cm]))
    e.append(Spacer(1, 4))
    e.append(P(
        '<b>Validation during fine-tuning.</b> Every 10 epochs (2,000 iterations) the current model super-resolves the '
        'first 20 validation images with the full ' + Tn + '-step reverse process, using the schedule of its own run '
        'and a fixed sampling seed, so every validation pass and both runs start from the same noise. We record mean '
        'PSNR and SSIM against the 128×128 ground truth and save all generated images. Validation was used for '
        'monitoring only: no hyperparameter and no checkpoint was selected with it. A checkpoint is written every 25 '
        'epochs, and the final one (epoch 100) is the model that is evaluated.'))
    e.append(P(
        '<b>Test evaluation.</b> The 100 test images were touched only after fine-tuning had finished. Each run '
        'super-resolves all of them once with the full ' + Tn + '-step reverse process, in batches of 25, starting from '
        'the same seeded Gaussian noise in all three runs. We report PSNR, SSIM and LPIPS (AlexNet) per image against '
        'the ground truth, and compare runs with paired per-image differences. Bicubic upsampling is included as a '
        'reference.'))
    e.append(P(
        '<b>Hardware and software.</b> One {} ({} GB), {}. Python {}, PyTorch {} (CUDA {}), torchvision {}, NumPy {}, '
        'OpenCV {}, Pillow {}, lpips {}, tensorboardX {}, wandb {}.'.format(
            env['gpu'], env['gpu_memory_gb'], env['os'].replace('-', ' ', 1).split('-')[0], env['python'],
            env['torch'], env['cuda_runtime'], env['torchvision'], env['numpy'], env['opencv-python'], env['Pillow'],
            env['lpips'], env['tensorboardX'], env['wandb'])))

    # ============================================================ 5
    e.append(P('5. Implementation', H2))
    e.append(P(
        '<b>Files and lines changed.</b> Six files of the original codebase were modified. Every changed line carries '
        'a ' + code('# MODIFIED') + ' comment and every added line an ' + code('# ADDED') + ' comment. Line numbers '
        'below refer to the submitted files.'))
    purpose = {
        'sr.py': 'Named constant ' + code('SCHEDULE_EXPONENT_X = 2.50') + ' at the top and the function that applies '
                 'it; random seed; validation in batches with a fixed sampling seed; SSIM during validation.',
        'model/sr3_modules/diffusion.py': 'The schedule change itself: ' + AB + ' is raised to the power x and '
                                          'β<sub>t</sub> is re-derived; bound assertions; batched sampling returns the whole batch.',
        'model/model.py': 'Schedule is rebuilt from the config after a checkpoint is loaded; option to start from '
                          'weights only; the active schedule is written to the log.',
        'core/logger.py': 'Number of validation images during training is configurable.',
        'core/wandb_logger.py': 'Run name, group and job type are passed to Weights &amp; Biases.',
        'data/__init__.py': 'Validation batch size is configurable.',
    }
    rows = [['File', 'Changed lines (submitted file)', 'Purpose']]
    for f in MODIFIED_FILES:
        _, _, hunks = file_hunks(f)
        rows.append([code(f), ranges([(j1 + 1, j2) for _, _, j1, j2 in hunks if j2 > j1]), purpose[f]])
    e.append(table(rows, [4.0 * cm, 3.9 * cm, W - 7.9 * cm]))
    e.append(Spacer(1, 4))
    e.append(P(
        'Eight new scripts were added under ' + code('assignment/') + ' (data preparation, schedule analysis, test '
        'evaluation, log collection, run comparison, and three builders for the deliverables and the two PDFs) together with '
        'three run configs. The two fine-tuning configs differ only in the run name and in the flag '
        + code('"use_power_schedule"') + ', which is false for Run A and true for Run B.'))
    e.append(P(
        '<b>How the transformation is implemented.</b> The original code builds β<sub>t</sub>, then α<sub>t</sub> and '
        + AB + ', and derives every other buffer from them. We raise ' + AB + ' to the power x right after it is '
        'computed and then recompute β<sub>t</sub> = 1 − ' + AB + '<super>x</super> / ᾱ<sub>t−1</sub><super>x</super> '
        'and α<sub>t</sub>, so that the posterior mean and variance used in sampling stay consistent with the modified '
        'forward process. The training loss needs no change, because it already draws its noise level from the '
        + AB + ' table.'))
    e.append(P(
        '<b>A pitfall we had to fix.</b> The checkpoint file contains the schedule buffers of the run that wrote it, '
        'and the original training path never rebuilds them after loading. Left alone, Run B would have trained with '
        'the modified noise levels while holding native buffers. ' + code('model/model.py') + ' now rebuilds the '
        'schedule from the config immediately after loading.'))
    e.append(P(
        '<b>Note on the named constant.</b> When Run A and Run B were executed, the exponent was read from the config '
        'entry ' + code('"theta_power": 2.5') + '. To meet the submission requirement we afterwards moved it into the '
        'constant ' + code('SCHEDULE_EXPONENT_X') + ' at the top of ' + code('sr.py') + ' and changed the comment '
        'markers to the required spelling. We then rebuilt both schedules through the submitted code and compared '
        'them with the buffers stored in the two fine-tuned checkpoints: they are bit-for-bit identical, so the '
        'submitted code produces exactly the schedules the runs used.'))
    e.append(P('<b>Original and modified lines, side by side.</b> Red: original lines that were replaced or removed. '
               'Green: lines in the submitted file. Unchanged lines are omitted.'))
    for f in MODIFIED_FILES:
        e.append(side_by_side(f))
        e.append(Spacer(1, 6))
    e.append(P('<b>How we verified that Run B used {' + TH + '<super>x</super>} and not {' + TH + '}.</b> Four '
               'independent checks, each of which distinguishes the two schedules:'))
    vb, va = ver['Run B'], ver['Run A']
    e.append(table([
        ['Check', 'Run A', 'Run B'],
        ['Training log, line "Noise schedule [train]" written when training starts',
         code('theta_power=1.00') + ', ᾱ<sub>T</sub> = ' + sci(va['stored_alpha_bar_T']),
         code('theta_power=2.50') + ', ᾱ<sub>T</sub> = ' + sci(vb['stored_alpha_bar_T'])],
        [code('alphas_cumprod') + ' buffer saved inside the final checkpoint, compared in double precision with '
         + AB + ' and with ' + AB + '<super>' + x + '</super>',
         'equals ' + AB + ': {}; equals '.format('yes' if va['stored_buffer_matches_native'] else 'no') + AB
         + '<super>' + x + '</super>: {}'.format('yes' if va['stored_buffer_matches_native_pow_x'] else 'no'),
         'equals ' + AB + ': {}; equals '.format('yes' if vb['stored_buffer_matches_native'] else 'no') + AB
         + '<super>' + x + '</super>: {}'.format('yes' if vb['stored_buffer_matches_native_pow_x'] else 'no')],
        ['Schedule recorded by the test evaluation (' + code('eval_config.json') + ')',
         'ᾱ<sub>1</sub> = {:.7f}, ᾱ<sub>T</sub> = {}'.format(va['stored_alpha_bar_1'], sci(va['test_eval_alpha_bar_T'])),
         'ᾱ<sub>1</sub> = {:.7f}, ᾱ<sub>T</sub> = {}'.format(vb['stored_alpha_bar_1'], sci(vb['test_eval_alpha_bar_T']))],
        ['Reverse-process snapshots at t = 1,000 (Figure 6)',
         'image structure already visible (' + AB + ' = {:.3f})'.format(sch['theta_at']['1000']['native']),
         'still indistinguishable from noise (' + AB + '<super>x</super> = {:.4f})'.format(sch['theta_at']['1000']['power'])],
    ], [6.4 * cm, (W - 6.4 * cm) / 2, (W - 6.4 * cm) / 2]))
    e.append(P('Table 2. Evidence that the two fine-tuning runs used different schedules. The expected values are '
               'ᾱ<sub>T</sub> = {} for the native schedule and {} for x = {}.'.format(
                   sci(sch['theta_range_native'][1]), sci(sch['theta_range_power'][1]), x), CAP))
    e.append(P(
        'In addition, the schedule produced by the model code is compared against an independent double-precision '
        'computation in ' + code('assignment/schedule_analysis.py') + ', and the two agree for both exponents.'))

    # ============================================================ 6
    e.append(P('6. Results', H2))
    e.append(P('<b>Run 0 outputs.</b> Figure 2 shows the pretrained model applied directly to eight test images.'))
    e.append(figure('run0_samples.png',
                    'Figure 2. Run 0. Rows: 16×16 input (enlarged), bicubic upsampling, Run 0 output, ground truth. '
                    'All 100 outputs are in assignment_results/run_0_baseline/test_outputs/sr/.'))
    e.append(P(
        '<b>Fine-tuning loss.</b> The script logged the loss of one batch every 100 iterations, that is two batches '
        'per epoch. The value in the submitted CSV files (' + code('epoch, train_loss') + ') is the mean of those two '
        'batches, not the mean over all 200 batches of the epoch, which was not recorded. Because the loss of a batch '
        'depends strongly on the timestep it happened to draw, the per-epoch values are noisy, and Figure 3 adds a '
        '10-epoch moving average.'))
    e.append(figure('loss_curves.png',
                    'Figure 3. Fine-tuning loss of Run A and Run B on one plot (log scale). Thin lines: per-epoch '
                    'values from the CSV files. Thick lines: 10-epoch moving average.', width=W * 0.86))
    e.append(P(
        'The two curves have the same shape because both runs share a seed and therefore draw the same timestep index '
        'in every iteration. Run B\'s averaged curve lies below Run A\'s throughout (mean over all epochs {:.3f} against {:.3f}). This does '
        'not mean Run B learned better: at the same timestep index its input is noisier, and predicting the noise in a '
        'noisier image is an easier problem. Loss values are comparable within a run, not across the two schedules.'
        .format(loss['Run B']['mean_all_epochs'], loss['Run A']['mean_all_epochs'])))
    e.append(P(
        '<b>Samples at five fine-tuning checkpoints.</b> Epochs 20, 40, 60, 80 and 100. The samples are the ones '
        'generated by the model at that epoch during validation, for the same three validation images in both runs.'))
    e.append(figure('checkpoint_samples_runA.png', 'Figure 4. Run A (native schedule), samples at epochs 20 to 100.'))
    e.append(figure('checkpoint_samples_runB.png',
                    'Figure 5. Run B (schedule raised to x = ' + x + '), samples at epochs 20 to 100.'))
    rows = [['Epoch', 'PSNR Run A', 'PSNR Run B', 'B − A (± s.e.)', 'LPIPS Run A', 'LPIPS Run B', 'B − A (± s.e.)']]
    for ep in num['sample_epochs']:
        r = val[ep]
        rows.append([ep, '{:.2f}'.format(r['runA_psnr']), '{:.2f}'.format(r['runB_psnr']),
                     '{:+.2f} ± {:.2f}'.format(r['diff_psnr'], r['se_diff_psnr']),
                     '{:.3f}'.format(r['runA_lpips']), '{:.3f}'.format(r['runB_lpips']),
                     '{:+.3f} ± {:.3f}'.format(r['diff_lpips'], r['se_diff_lpips'])])
    e.append(table(rows, [1.5 * cm] + [(W - 1.5 * cm) / 6] * 6, center_from=0))
    e.append(P('Table 3. The two runs at the five checkpoints, on the 20 validation images sampled during '
               'fine-tuning. Differences are paired per image; s.e. is the standard error of the mean difference.', CAP))
    e.append(P(
        '<b>Inference trajectories.</b> Noisy sample x<sub>t</sub> at t = 1800, 1400, 1000, 600 and 200 during the '
        'reverse process of the final models, for three test images. The snapshots come from the same sampling runs '
        'that produced the test scores.'))
    e.append(figure('trajectory_runA.png', 'Figure 6a. Run A, final model: x_t at five equally spaced timesteps.'))
    e.append(figure('trajectory_runB.png', 'Figure 6b. Run B, final model: x_t at the same timesteps and from the '
                                           'same starting noise.'))
    e.append(P(
        '<b>Test-set scores.</b> Our chosen metric is <b>PSNR</b>: in a medical image, pixel-level fidelity to the '
        'true frame matters more than a plausible-looking texture, and PSNR measures exactly that. SSIM and LPIPS are '
        'reported alongside it.'))
    b = summ['Run 0']
    rows = [['Method', 'PSNR (dB) ↑', 'SSIM ↑', 'LPIPS ↓'],
            ['Bicubic upsampling (reference)', pm(b['bicubic_psnr_mean'], b['bicubic_psnr_std'], 2),
             pm(b['bicubic_ssim_mean'], b['bicubic_ssim_std'], 3), pm(b['bicubic_lpips_mean'], b['bicubic_lpips_std'], 3)]]
    for label, name in (('Run 0', 'Run 0: pretrained, no fine-tuning'),
                        ('Run A', 'Run A: fine-tuned, ' + AB),
                        ('Run B', 'Run B: fine-tuned, ' + AB + '<super>' + x + '</super>')):
        s = summ[label]
        rows.append([name, '<b>' + pm(s['psnr_mean'], s['psnr_std'], 2) + '</b>', pm(s['ssim_mean'], s['ssim_std'], 3),
                     pm(s['lpips_mean'], s['lpips_std'], 3)])
    e.append(table(rows, [6.4 * cm] + [(W - 6.4 * cm) / 3] * 3, center_from=1))
    e.append(P('Table 4. Test-set scores, mean ± standard deviation over the 100 test images. One sample per image, '
               + Tn + ' sampling steps, same starting noise in all runs.', CAP))
    rows = [['Comparison', 'PSNR difference (dB)', 'Images improved', 'SSIM difference', 'LPIPS difference']]
    for name, d in (('Run A − Run 0', a0), ('Run B − Run 0', b0), ('Run B − Run A', ab)):
        rows.append([name, '{:+.2f} ± {:.2f}'.format(d['psnr']['mean_difference'], d['psnr']['standard_error']),
                     '{} / 100'.format(d['psnr']['images_where_second_run_is_better']),
                     '{:+.3f} ± {:.3f}'.format(d['ssim']['mean_difference'], d['ssim']['standard_error']),
                     '{:+.3f} ± {:.3f}'.format(d['lpips']['mean_difference'], d['lpips']['standard_error'])])
    e.append(table(rows, [3.4 * cm] + [(W - 3.4 * cm) / 4] * 4, center_from=1))
    e.append(P('Table 5. Paired per-image differences on the test set, mean ± standard error. "Images improved" '
               'counts the images on which the first-named run has the higher PSNR.', CAP))
    e.append(figure('three_runs.png', 'Figure 7. The three runs on the same eight test images.', width=W * 0.92))

    # ============================================================ 7
    e.append(P('7. Analysis', H2))
    e.append(P(
        '<b>Which run is better.</b> On the chosen metric the native schedule wins. Run A reaches {:.2f} dB and Run B '
        '{:.2f} dB, a paired difference of {:+.2f} ± {:.2f} dB, and Run B has the higher PSNR on only {} of the 100 '
        'test images. SSIM agrees ({:+.3f} ± {:.3f}). LPIPS does not separate the runs at all ({:+.3f} ± {:.3f}, each '
        'run better on {} images). So Run B\'s outputs look as realistic as Run A\'s but are less faithful to the true '
        'image.'.format(summ['Run A']['psnr_mean'], summ['Run B']['psnr_mean'], ab['psnr']['mean_difference'],
                        ab['psnr']['standard_error'], ab['psnr']['images_where_second_run_is_better'],
                        ab['ssim']['mean_difference'], ab['ssim']['standard_error'], ab['lpips']['mean_difference'],
                        ab['lpips']['standard_error'], ab['lpips']['images_where_second_run_is_better'])))
    e.append(P(
        '<b>When the difference became visible.</b> By eye, the samples of the two runs in Figures 4 and 5 are hard to '
        'tell apart at any of the five checkpoints. The numbers in Table 3 say the same about fidelity: at epochs 20, '
        '40, 60 and 80 the PSNR difference is within about one standard error of zero and changes sign, and only at '
        '<b>epoch 100</b> does it exceed the noise ({:+.2f} ± {:.2f} dB on validation, matching the test result). '
        'Perceptual quality behaves differently. LPIPS already favours Run A at the first sampled checkpoint, epoch 20 '
        '({:+.3f} ± {:.3f}), and the gap was largest at the very first validation pass at epoch 10 (LPIPS {:.3f} for '
        'Run B against {:.3f} for Run A). It then shrinks, and at epoch 100 the two runs are level. Our reading is that '
        'Run B needed longer to adapt, and that the PSNR gap is something we can only establish at the final '
        'checkpoint.'.format(val[100]['diff_psnr'], val[100]['se_diff_psnr'], val[20]['diff_lpips'],
                             val[20]['se_diff_lpips'], val[10]['runB_lpips'], val[10]['runA_lpips'])))
    e.append(P('<b>Mechanistic explanation.</b> SR3 conditions on the noise level, not on the timestep index. Changing '
               'the schedule therefore does not give the network a different task; it changes how training and '
               'sampling effort is distributed over noise levels. Three consequences follow from Section 3.'))
    for item in [
        '<b>Wasted training steps.</b> Training draws t uniformly. Under ' + AB + '<super>' + x + '</super>, {:.1f}% of '
        'the draws land at t ≥ {}, where the SNR is below {} and x<sub>t</sub> is, for practical purposes, pure noise. '
        'The best possible prediction there is the input itself, so these updates teach the network nothing about '
        'endoscopy images. Run A spends all of its 20,000 updates inside the range where the image still matters; '
        'Run B spends about {:,} of them there.'.format(
            100 * sch['fraction_below_native_terminal_level'], t_cross, sci(sch['snr_at']['2000']['native'], 1),
            int(round(20000 * (1 - sch['fraction_below_native_terminal_level'])))),
        '<b>Less practice on fine detail.</b> Fine texture is decided at low noise. Timesteps with more signal than '
        'noise make up {:.1f}% of the native schedule ({} of {}) but only {:.1f}% of the modified one ({}). Run B '
        'therefore gets roughly {:.0f}% as many updates on the noise levels that fix the final pixel values.'.format(
            100 * sch['timesteps_with_snr_above_1']['native'] / T, sch['timesteps_with_snr_above_1']['native'], Tn,
            100 * sch['timesteps_with_snr_above_1']['power'] / T, sch['timesteps_with_snr_above_1']['power'],
            100 * sch['timesteps_with_snr_above_1']['power'] / sch['timesteps_with_snr_above_1']['native']),
        ('<b>Coarser sampling.</b> The reverse process of Run B spends its first {} steps where nothing can be decided '
         'and then has to cross the whole informative range in the remaining {:,} steps, with per-step variance about '
         '{} times larger. Figure 6 shows this directly: at t = 1,000 Run A already shows the layout of the image '
         '(' + AB + ' = {:.3f}) while Run B is still noise ({:.4f}); at t = 600 Run A is well formed ({:.2f}) and Run B '
         'is only starting ({:.2f}). The image is assembled in fewer, larger steps.').format(
            below, T - below, x, sch['theta_at']['1000']['native'], sch['theta_at']['1000']['power'],
            sch['theta_at_trajectory_timesteps']['600']['native'],
            sch['theta_at_trajectory_timesteps']['600']['power']),
    ]:
        e.append(P('• ' + item, LI))
    e.append(P(
        'The error pattern is consistent with this account. We split each test output\'s error into a smooth part and '
        'a detail part (Gaussian low-pass, σ = 3 pixels, and the remainder). Run B has more error than Run A in both: '
        '{:.0f} against {:.0f} in the smooth part and {:.0f} against {:.0f} in the detail part (mean squared error on '
        'the 0-255 scale). At the same time its outputs contain more fine detail, {:.0f}% of the ground truth\'s '
        'high-frequency energy against {:.0f}% for Run A. Run B produces at least as much texture, but less of it is in '
        'the right place, which is what equal LPIPS with lower PSNR looks like. A shift in overall colour is not the '
        'cause: it accounts for about 1% of the error in every run.'.format(
            dec['Run B']['low_freq_mse'], dec['Run A']['low_freq_mse'], dec['Run B']['high_freq_mse'],
            dec['Run A']['high_freq_mse'], 100 * dec['Run B']['high_freq_energy_ratio_vs_ground_truth'],
            100 * dec['Run A']['high_freq_energy_ratio_vs_ground_truth'])))
    e.append(P(
        '<b>How firm this is.</b> Each run was trained once. Validation PSNR moved by up to a decibel between '
        'neighbouring checkpoints in both runs (Run A fell to {:.2f} dB at epoch 50; Run B fell from {:.2f} dB at '
        'epoch 90 to {:.2f} dB at epoch 100). We evaluated the epoch-100 checkpoint of both runs because that was '
        'fixed in advance, but part of the {:.2f} dB test gap may be this checkpoint-to-checkpoint variation rather '
        'than the schedule. The direction of the effect agrees with the mechanism above; its size would need repeated '
        'seeds to pin down.'.format(val[50]['runA_psnr'], val[90]['runB_psnr'], val[100]['runB_psnr'],
                                    -ab['psnr']['mean_difference'])))
    e.append(P(
        '<b>Run 0 compared with the fine-tuned runs.</b> Without fine-tuning, the face-trained model scores {:.2f} dB, '
        'below plain bicubic upsampling ({:.2f} dB), yet its LPIPS ({:.3f}) is far better than bicubic ({:.3f}). It has '
        'learned how to sharpen: its outputs carry {:.0f}% of the ground truth\'s high-frequency energy, where bicubic '
        'carries {:.0f}%. But the detail it adds comes from faces. In Figure 2 the mucosa is rendered with smooth, '
        'skin-like surfaces and invented creases, and fine vessel patterns are missing, so the added detail raises the '
        'pixel error instead of lowering it. Fine-tuning corrects exactly this. Run A improves on Run 0 by '
        '{:+.2f} ± {:.2f} dB with {} of 100 images better, and Run B by {:+.2f} ± {:.2f} dB; LPIPS falls from {:.3f} to '
        '{:.3f} in both, with nearly every image improved.'.format(
            summ['Run 0']['psnr_mean'], summ['Run 0']['bicubic_psnr_mean'], summ['Run 0']['lpips_mean'],
            summ['Run 0']['bicubic_lpips_mean'], 100 * dec['Run 0']['high_freq_energy_ratio_vs_ground_truth'],
            100 * dec['Bicubic']['high_freq_energy_ratio_vs_ground_truth'], a0['psnr']['mean_difference'],
            a0['psnr']['standard_error'], a0['psnr']['images_where_second_run_is_better'],
            b0['psnr']['mean_difference'], b0['psnr']['standard_error'], summ['Run 0']['lpips_mean'],
            summ['Run A']['lpips_mean'])))
    e.append(P(
        'What this says about fine-tuning on an out-of-distribution dataset: the general skill of the pretrained model, '
        'turning a blurry input into a sharp image that agrees with it, carries over to a domain it has never seen. '
        'What does not carry over is its idea of what the detail should look like, and 800 images and 100 epochs are '
        'enough to replace that. The gain from fine-tuning ({:.2f} dB for Run A) is more than twice the difference '
        'between the two schedules ({:.2f} dB), so here adapting to the domain mattered more than the choice of '
        'schedule. Even after fine-tuning, SSIM stays below bicubic ({:.3f} against {:.3f}). At 8× from a 16×16 '
        'input the problem is heavily under-determined, and a single sharp sample is penalised by pixel-wise metrics '
        'relative to a blurry average.'.format(a0['psnr']['mean_difference'], -ab['psnr']['mean_difference'],
                                                summ['Run A']['ssim_mean'], summ['Run 0']['bicubic_ssim_mean'])))

    # ============================================================ 8
    e.append(P('8. Takeaway', H2))
    for item in [
        'The power transformation is not a small perturbation. For this schedule, raising ' + AB + ' to x = ' + x
        + ' is the same as running the native process {:.2f} times faster: about {:.0f}% of the timesteps end up at '
        'noise levels the pretrained model never saw and that carry no information about the image.'.format(
            sch['x'] ** 0.5, 100 * sch['fraction_below_native_terminal_level']),
        'The model coped with this better than we expected. Because SR3 is conditioned on the noise level, the '
        'modified schedule reuses what was learned in pretraining instead of invalidating it, and Run B still '
        'improved clearly on Run 0. The cost showed up in fidelity ({:.2f} dB) and not in perceptual quality.'.format(
            -ab['psnr']['mean_difference']),
        'A lower training loss did not mean a better model. Run B\'s loss was lower throughout only because its '
        'denoising problems were easier on average. Loss curves under different schedules cannot be compared directly.',
        'Fine-tuning mattered more than the schedule. A model trained only on faces transferred its sharpening '
        'ability to endoscopy at once, but needed fine-tuning to stop producing face-like texture.',
        'Checking the mechanics was worth the effort. The checkpoint silently carries its own schedule, and without '
        'the fix in ' + code('model.py') + ' and the checks in Table 2 we could not have been sure which schedule '
        'Run B really used.',
        'One seed per run and one sample per image leave real uncertainty. Validation scores swung by a decibel '
        'between checkpoints, so with more time we would repeat both runs with several seeds and average over '
        'several samples before making a stronger claim about the size of the gap.',
    ]:
        e.append(P('• ' + item, LI))

    # ============================================================ 9
    e.append(P('9. Contribution Statement', H2))
    rows = [['Member', 'Roll number', 'Responsible for', 'Signature']]
    for name, roll in MEMBERS:
        rows.append([name, roll, RESPONSIBILITY, '<br/><br/>'])
    e.append(table(rows, [3.6 * cm, 2.6 * cm, W - 10.2 * cm, 4.0 * cm]))
    e.append(Spacer(1, 4))
    e.append(P('We confirm that this report describes our own work and that the contributions listed above are '
               'accurate.'))

    # ============================================================ appendix
    e.append(P('Appendix: submission contents', H2))
    e.append(table([
        ['Component', 'Location in the repository'],
        ['Modified code and README', 'repository root, branch ' + code('feature/phase2-report') + '; exponent constant in '
                                     + code('sr.py') + ', line 3; reproduction commands, seed and library versions in ' + code('README.md')],
        ['Fine-tuning logs', code('assignment_results/phase2/finetuning_logs/runA_train_loss.csv') + ' and '
                             + code('runB_train_loss.csv') + '; full logs in ' + code('assignment_results/run_*/logs/')],
        ['Checkpoint samples', code('assignment_results/phase2/checkpoint_samples/') + ', files named '
                               + code('runA_epoch020_sample1.png') + ' etc., each with a caption bar'],
        ['Inference trajectories', code('assignment_results/phase2/inference_trajectories/') + ', files named '
                                   + code('runB_t1400_sample2.png') + ' etc.; all eleven snapshots per image in '
                                   + code('assignment_results/run_*/trajectories/')],
        ['Run 0 outputs', code('assignment_results/run_0_baseline/test_outputs/sr/') + ' (100 images)'],
        ['Scores', code('metrics_per_image.csv') + ' and ' + code('metrics_summary.json') + ' in each run folder; '
                   + code('assignment_results/comparison/')],
    ], [3.6 * cm, W - 3.6 * cm]))

    doc.build(e)
    print('wrote', os.path.join(P2, 'Phase2_Report.pdf'))


if __name__ == '__main__':
    main()
