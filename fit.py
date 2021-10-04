from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.optim import Adam
from tqdm import trange


def parse_args():
    '''
    Parse arguments.
    '''
    x = ArgumentParser()
    x.add_argument('--state', type=str, required=True)
    x.add_argument('--device', type=str, default='cpu')
    x.add_argument('--in_counties', type=str, default='data/counties.csv')
    x.add_argument('--out_dir', type=str, default='data/')
    x.add_argument('--im_side', type=int, default=1000)
    x.add_argument('--num_districts', type=int, default=16)
    x.add_argument('--num_epochs', type=int, default=1000)
    x.add_argument('--batches_per_epoch', type=int, default=2000)
    x.add_argument('--power', type=float, default=0.7)
    return x.parse_args()


def load(f):
    '''
    Load data file.
    '''
    state2counties = defaultdict(list)
    f = open(f)
    next(f)
    for line in f:
        state, fips, lat, lon, dem, rep = line.strip().split(',')
        fips = int(fips)
        lat = float(lat)
        lon = float(lon)
        dem = int(dem)
        rep = int(rep)
        county = fips, lat, lon, dem, rep
        state2counties[state].append(county)
    return state2counties


def load_state(f, state, device):
    '''
    Load data for the given state.
    '''
    state2counties = load(f)
    counties = state2counties[state]
    fips, lat, lon, dem, rep = zip(*counties)
    fips = torch.tensor(fips, dtype=torch.int32, device=device)
    lat = torch.tensor(lat, dtype=torch.float32, device=device)
    lon = torch.tensor(lon, dtype=torch.float32, device=device)
    dem = torch.tensor(dem, dtype=torch.float32, device=device)
    rep = torch.tensor(rep, dtype=torch.float32, device=device)
    return fips, lat, lon, dem, rep


def norm_coords(lat, lon):
    '''
    Normalize coordinates to be from zero to one on the smaller axis.

    For example, Virginia to 0 <= lat <= 1, 0 <= lon <= 2.81.

    In:
        lat: -90 <= lat <= 90
        lon: -180 <= lat <= 180

    Out:
        lat: 0 <= lat <= 1 or ratio
        lon: 0 <= lon <= ratio or 1
    '''
    lat_span = max(lat) - min(lat)
    lon_span = max(lon) - min(lon)
    min_span = min(lat_span, lon_span)
    lat = (lat - lat.min()) / lat_span
    lat = lat * lat_span / min_span
    lon = (lon - lon.min()) / lon_span
    lon = lon * lon_span / min_span
    return lat, lon


def dump_stats(s, x):
    '''
    Dump stats.
    '''
    print('%s: min %7.3f mean %7.3f std %7.3f max %7.3f' %
          (s, x.min(), x.mean(), x.std(), x.max()))


def get_pop_loss(c2d, c_pops):
    '''
    Loss: want balanced district populations.
    '''
    d_pops = torch.einsum('cd,c->d', [c2d, c_pops])
    fracs = d_pops / d_pops.sum()
    return fracs.std() * 2


def get_lean_loss(c2d, c_dems, c_reps):
    '''
    Loss: want balanced district political leans.
    '''
    d_dems = torch.einsum('cd,c->d', [c2d, c_dems])
    d_reps = torch.einsum('cd,c->d', [c2d, c_reps])
    d_leans = d_reps / (d_dems + d_reps) * 2 - 1
    fracs = d_leans / d_leans.sum()
    return fracs.std() * 2


def get_spread_loss(c2d, c_lats, c_lons):
    '''
    Loss: want contiguous (not Gerrymandered) district shapes.
    '''
    num_c = len(c2d)
    d_lats = torch.einsum('cd,c->d', [c2d, c_lats]) / num_c
    d_lons = torch.einsum('cd,c->d', [c2d, c_lons]) / num_c
    lat_dist = c_lats.unsqueeze(1) - d_lats.unsqueeze(0)
    lon_dist = c_lons.unsqueeze(1) - d_lons.unsqueeze(0)
    dist = lat_dist ** 2 + lon_dist ** 2
    dist = dist.tanh()
    return 2 * torch.einsum('cd,cd->', [c2d, dist]) / c2d.numel()


def get_sharp_loss(c2d):
    '''
    Want clearly defined districts.
    '''
    r = 1 - c2d.max(1).values.mean()
    return torch.zeros(1, device=c2d.device)


def mean(x):
    '''
    Mean.
    '''
    return sum(x) / len(x)


def draw_map(c2d, c_lats, c_lons, c_colors, im_side, device, power, out):
    '''
    Render a map of a state's districts.
    '''
    c2d = c2d.to(device)
    c_lats = c_lats.to(device)
    c_lons = c_lons.to(device)
    c_colors = c_colors.to(device)

    im_height = int(c_lats.max() * im_side)
    im_width = int(c_lons.max() * im_side)

    im_lats = torch.linspace(0, c_lats.max(), im_height, device=device)
    im_lons = torch.linspace(0, c_lons.max(), im_width, device=device)

    lat_dist = im_lats.unsqueeze(0) - c_lats.unsqueeze(1)
    lon_dist = im_lons.unsqueeze(0) - c_lons.unsqueeze(1)
    dist = (lat_dist.unsqueeze(2) ** 2 + lon_dist.unsqueeze(1) ** 2).sqrt()
    dist = dist / dist.mean()
    heatmap = 1 / dist ** power
    heatmap = heatmap.softmax(0)

    num_c = len(c_lats)
    im = torch.einsum('nc,nhw->chw', [c_colors, heatmap])
    im = im.permute(1, 2, 0)
    im = (im * 255).type(torch.uint8)
    im = im.detach().cpu().numpy()
    im = np.flip(im, 0)
    im = Image.fromarray(im)
    im.save(out)


def main(args):
    '''
    Train.
    '''
    # Init device.
    device = torch.device(args.device)
    print('Device: %s' % args.device)

    # Load dataset of counties and votes.
    c_fipss, c_lats, c_lons, c_dems, c_reps = load_state(
        args.in_counties, args.state, device)
    c_pops = c_dems + c_reps
    c_leans = c_reps / c_pops * 2 - 1
    dump_stats('Orig lat', c_lats)
    dump_stats('Orig lon', c_lons)

    # Normalize county locations onto a grid.
    c_lats, c_lons = norm_coords(c_lats, c_lons)
    dump_stats('Norm lat', c_lats)
    dump_stats('Norm lon', c_lons)
    c_lats, c_lons = norm_coords(c_lats, c_lons)

    # Init model, ie mapping of counties to districts.
    num_c = len(c_lats)
    log_c2d = 0.05 * torch.randn(num_c, args.num_districts, device=device)
    log_c2d.requires_grad_(True)
    opt = Adam([log_c2d])

    # For visualization.
    d_colors = torch.rand(args.num_districts, 3, device=device)

    # Fit.
    for epoch in range(args.num_epochs):
        ee_pop = []
        ee_lean = []
        ee_spread = []
        ee_sharp = []
        ee = []
        for batch in trange(args.batches_per_epoch, leave=False):
            opt.zero_grad()
            c2d = log_c2d.softmax(1)
            e_pop = get_pop_loss(c2d, c_pops)
            e_lean = get_lean_loss(c2d, c_dems, c_reps)
            e_spread = get_spread_loss(c2d, c_lats, c_lons)
            e_sharp = get_sharp_loss(c2d)
            e = e_pop + e_lean + e_spread + e_sharp
            e.backward()
            opt.step()
            ee_pop.append(e_pop.item())
            ee_lean.append(e_lean.item())
            ee_spread.append(e_spread.item())
            ee_sharp.append(e_sharp.item())
            ee.append(e.item())
            if not batch:
                c_colors = torch.einsum('cd,dt->ct', [c2d, d_colors])
                cpu = torch.device('cpu')
                out = '%s/%s_epoch_%05d.jpg' % (args.out_dir, args.state.lower(), epoch)
                draw_map(c2d, c_lats, c_lons, c_colors, args.im_side, cpu, args.power, out)
        e_pop = mean(ee_pop)
        e_lean = mean(ee_lean)
        e_spread = mean(ee_spread)
        e_sharp = mean(ee_sharp)
        e = mean(ee)
        print('%6d | e %8.6f | pop %8.6f lean %8.6f spread %8.6f sharp %8.6f' %
              (epoch, e, e_pop, e_lean, e_spread, e_sharp))


if __name__ == '__main__':
    main(parse_args())
