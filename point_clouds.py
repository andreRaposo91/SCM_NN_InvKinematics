import numpy as np
from itertools import chain
from scipy.spatial.transform import Rotation as R
from math import ceil

def generate_random_points(min_val, max_val, num_points, min_distance):
    points = np.zeros((num_points, 3), dtype=int) #[(0, 0, 0)] * num_points

    bad = 0
    opt1 = 0
    opt2 = 0
    opt3 = 0
    i = 0

    def diff(point):
        return np.abs([
            point[0] - point[1],
            point[0] - point[2],
            point[1] - point[2],
            ])

    while i < num_points:
        new_point = np.random.randint(min_val, max_val, size=3)
        
        # Skip point if reaches curvature limit of spring
        if ((new_point[0] + new_point[1]) > 4200 and new_point[2] < 1220) or \
           ((new_point[1] + new_point[2]) > 4200 and new_point[0] < 1220)  or \
           ((new_point[2] + new_point[0]) > 4200 and new_point[1] < 1220):
        # if any(diff(new_point) > 900):
            # print(new_point)
            opt3 += 1
            # max_curv.append(new_point)
            continue

        # Check if next point is far from previous
        if i > 0 and np.linalg.norm(np.array(new_point) - np.array(points[i-1])) >= min_distance:
            points[i] = new_point
            opt1 += 1
            i += 1
        # If close, check if the previous point is far from other points (except the one before)
        elif i > 0:
            far_from_others = all(
                np.linalg.norm(np.array(new_point) - np.array(existing_point)) >= min_distance
                for existing_point in points[:i - 1]
            )
            if far_from_others:
                points[i] = new_point
                opt2 += 1
                i += 1
        # First point
        elif i == 0:
            points[i] = new_point
            opt3 += 1
            i += 1
        else:
            bad += 1

    # print(f'far from previous: {opt1}, \nprevious far from others: {opt2}, \nfirst: {opt3}, \nbad: {bad}')
    # print(max_curv[np.argmax(np.sum(max_curv, axis=0), axis=0)])
    # print(max_curv[np.argmin(np.sum(max_curv, axis=0), axis=0)])
    return points

# def vert_up_down_seq(min_val, max_val, num=10, reps=3):
def vert_up_down_seq(min_val, max_val, nums=[10]):
    traj = np.zeros((sum([num*2 for num in nums]), 3), dtype=int)
    offset = 0
    for num in nums:
        step = np.ceil((max_val - min_val) / num).astype(int)
        traj[offset:offset+num*2] = np.vstack([np.repeat([np.arange(min_val, max_val + step, step)], 3, axis=0).T,
                                                 np.flip(np.repeat([np.arange(min_val, max_val + step, step)], 3, axis=0).T[1:-1])])
        offset += num * 2
    return traj
    # return np.vstack([traj, np.flip(traj[1:-1])] * reps)

def vert_to_point_seq(min_val, max_val, num=10):
    step = (max_val - min_val) // num
    points = np.arange(min_val, max_val + step, step)
    len_pts = len(points)
    traj = np.array([list(chain(*zip([points[i]] * len_pts, np.delete(points, i)))) for i in range(len_pts)]).flatten()
    return np.vstack((traj, traj, traj)).T

def curv_up_down_seq(min_val, max_val, num=10, reps=3, REST=1220, servos=[0,1,2]):  
    # print(num)
    step = (max_val - min_val) // num
    traj1 = np.zeros(((num**2 + num) * 3, 3), dtype=int)
    # print(traj1.shape)
    traj2 = np.zeros(((num**2 + num) * 3, 3), dtype=int)
    if (max_val - min_val) % num == 0:
        targets = np.arange(min_val, max_val + step, step)
    else: targets = np.arange(min_val, max_val, step)
    # print(targets)
    # print(np.array([targets] * num).flatten().shape)
    # print(np.repeat(targets, num).shape)
    del_mask = np.zeros((len(traj1),), dtype=bool)
    for i in servos:
        idx = [0,1,2]
        idx.remove(i)
        # print(i, i*(num**2 + num), (i+1)*(num**2 + num))
        offset = num**2 + num
        traj1[i*(offset):(i+1)*(offset),i] = np.repeat(targets, num)
        traj1[i*(offset):(i+1)*(offset),idx] = np.repeat([np.array([targets] * num).flatten()], 2, axis=0).T
        
        traj2[i*(offset):(i+1)*(offset),idx[0]] = (traj1[i*(offset):(i+1)*(offset),idx[0]] + traj1[i*(offset):(i+1)*(offset),i]) / 2
        traj2[i*(offset):(i+1)*(offset),[i, idx[1]]] = np.array([traj1[i*(offset):(i+1)*(offset),i], traj1[i*(offset):(i+1)*(offset),idx[0]]]).T
        
        del_mask[i*(offset):(i+1)*(offset)] = (np.sum(traj1[i*(offset):(i+1)*(offset),idx], axis=1) > 4200) & (traj1[i*(offset):(i+1)*(offset),i] < REST)
    traj1 = traj1[~del_mask]
    traj2 = traj2[~del_mask]
    # print(np.vstack([traj2, np.flip(traj2[1:-1])] * reps)[:10]) # , traj2, np.flip(traj2[1:-1])
    return np.vstack([traj1, np.flip(traj1[1:-1]), traj2, np.flip(traj2[1:-1])] * reps) # 

def curv_test(min_val, max_val, move_num=12, fixed_pos=[800, 1220, 1800], servos=[0,], reps=2):
    REST = 1220
    MAX_CURV = 4200  
    # print(num)
    move_step = (max_val - min_val) // (move_num - 1)
    traj1 = np.zeros((move_num * len(fixed_pos) * len(servos) * 2 - len(servos), 3), dtype=int)
    traj2 = np.zeros((move_num * len(fixed_pos) * len(servos) * 2 - len(servos), 3), dtype=int)
    # move_pts = np.arange(min_val, max_val + move_step, move_step)
    # move_pts = np.arange(min_val, max_val, move_step)
    move_pts = np.linspace(min_val, max_val, move_num, dtype=int)
    # if (max_val - min_val) % move_num != 0:
        # move_pts = np.arange(min_val, max_val + move_step, move_step)
    # else: move_pts = np.arange(min_val, max_val, move_step)
    # fixed_pts = np.arange(min_val, max_val, fixed_step)
    # print(np.array([targets] * num).flatten().shape)
    # print(np.repeat(targets, num).shape)
    offset = move_num * len(fixed_pos) * 2 - 1
    temp_traj1 = np.zeros(((offset+1) // 2, 3))
    temp_traj2 = np.zeros(((offset+1) // 2, 3))
    del_mask = np.zeros((len(traj1),), dtype=bool)
    for i in servos:
        idx = [0,1,2]
        idx.remove(i)
        # traj1[i*offset:(i+1)*offset, i] = np.repeat(fixed_pos, move_num)
        # traj1[i*offset:(i+1)*offset, idx] = np.repeat([np.array([move_pts] * len(fixed_pos)).flatten()], 2, axis=0).T
        temp_traj1[:, i] = np.repeat(fixed_pos, move_num)
        temp_traj1[:, idx] = np.repeat([np.array([move_pts] * len(fixed_pos)).flatten()], 2, axis=0).T

        traj1[i*offset:(i+1)*offset] = np.concatenate((temp_traj1, np.flip(temp_traj1[:-1])), axis=0)

        
        # traj2[i*offset:(i+1)*offset, idx] = np.repeat([np.repeat(fixed_pos, move_num)], 2, axis=0).T
        # traj2[i*offset:(i+1)*offset, i] = np.array([move_pts] * len(fixed_pos)).flatten()
        
        temp_traj2[:, idx] = np.repeat([np.repeat(fixed_pos, move_num)], 2, axis=0).T
        temp_traj2[:, i] = np.array([move_pts] * len(fixed_pos)).flatten()
        traj2[i*offset:(i+1)*offset] = np.concatenate((temp_traj2, np.flip(temp_traj2[:-1])), axis=0)

        del_mask[i*offset:(i+1)*offset] = (np.sum(traj1[i*offset:(i+1)*offset,idx], axis=1) > MAX_CURV) & (traj1[i*offset:(i+1)*offset,i] < REST) | \
            (np.sum(traj1[i*offset:(i+1)*offset,idx], axis=1) > MAX_CURV) & (traj1[i*offset:(i+1)*offset,i] < REST)


    traj1 = traj1[~del_mask]
    traj2 = traj2[~del_mask]

    # traj1 = np.unique(traj1, axis=0)
    # traj2 = np.unique(traj2, axis=0)
    # return np.vstack([traj1, np.flip(traj1[1:-1]), traj2, np.flip(traj2[1:-1])] * reps) # 
    # return np.vstack([traj1, np.flip(traj1[1:-1]),] * reps)
    # return np.vstack([traj2, np.flip(traj2[1:-1]),] * reps)
    return np.vstack([traj1, traj2])

def grid(min_val, max_val, resolution, randomize=False):
    vals = np.linspace(min_val, max_val, resolution, dtype=int)

    x, y, z = np.meshgrid(vals, vals, vals, indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    condition = (
        ((points[:, 0] + points[:, 1] < 4200) | (points[:, 2] > 1220)) &
        ((points[:, 1] + points[:, 2] < 4200) | (points[:, 0] > 1220)) &
        ((points[:, 0] + points[:, 2] < 4200) | (points[:, 1] > 1220))
    )

    filtered_points = points[condition]
    if randomize:
        np.random.shuffle(filtered_points)
    
    return filtered_points

def add_neighbours(points, num_of_neighbours, min_dist, max_dist):
    if num_of_neighbours > 3:
        raise ValueError("# of neighbours must be <= 3")

    final_points = np.repeat(points, num_of_neighbours * 2 + 1, axis=0)

    if __name__ == "__main__":
        print("possible grid pts w/ neighbours:", len(final_points))
    # print(final_points[:10])
    is_point_good = lambda point: True if \
        ((point[0] + point[1] < 4200) or (point[2] > 1220)) and \
        ((point[1] + point[2] < 4200) or (point[0] > 1220)) and \
        ((point[0] + point[2] < 4200) or (point[1] > 1220)) and \
        min(point) >= 750 and max(point) <= 2250 \
        else False
    rotx = lambda t: np.array([
        [1, 0, 0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t), np.cos(t)]
        ])
    roty = lambda t: np.array([
        [np.cos(t), 0, np.sin(t)],
        [0, 1, 0],
        [-np.sin(t), 0, np.cos(t)]
        ])
    rotz = lambda t: np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
        ])
    
    # dirs = np.concatenate((np.eye(3, dtype=int), -np.eye(3, dtype=int)), axis=0)
    dirs = np.eye(3, dtype=int)

    fail_idx = []
    max_rot_tries = 5000
    tol = 1
    for i, point in enumerate(final_points[::num_of_neighbours*2+1]):
        # print("-", point)
        offset = i * (num_of_neighbours * 2 + 1)
        rot = []
        tries = 0
        while tries < max_rot_tries:
            tries += 1
            del_list = []
            angles = np.random.uniform(0, 2 * np.pi, size=(3,))
        # angles = np.zeros((3,), dtype=int)
            rot = dirs @ rotx(angles[0]) @ roty(angles[1]) @ rotz(angles[2])
            for r in range(len(rot)):
                if not is_point_good(point + (rot[r] * min_dist * tol).astype(int)) \
                    and not is_point_good(point + (rot[r] * (min_dist + max_dist) / 2).astype(int)):
                    # print((rot[r] * min_dist).astype(int))
                    del_list.append(r)
            rot = np.delete(rot, del_list, axis=0)
            if len(rot) == num_of_neighbours:
                break

        if tries == max_rot_tries:
            # print("bad point", i, "; offset", offset)
            # len_idx = len(fail_idx)
            fail_idx += [offset + c for c in range(num_of_neighbours*2)]
            # print(fail_idx[len_idx:])
            continue

        n_added = 0
        max_neighb_tries = 500
        while n_added < num_of_neighbours:
            c = 0
            new_dir = np.random.choice(range(len(rot))) # pyright: ignore
            # new_point = np.array(point) + rot[new_dir] * np.random.randint(min_dist, lims[new_dir % 3])
            new_point = np.array(point) + rot[new_dir] * np.random.randint(min_dist, max_dist) * np.random.choice([1, -1]) # pyright: ignore
            max_dist_adjusted = max_dist
            while not is_point_good(new_point):
                c += 1
                # print("point", i+1, "neighbour", j+1)
                if max_dist_adjusted <= min_dist or c >= max_neighb_tries:
                    # print(c)       
                    break
                new_point = np.array(point) + rot[new_dir] * np.random.randint(min_dist, max_dist_adjusted) * np.random.choice([1, -1]) # pyright: ignore
                # max_dist_adjusted = max_dist // (c // 10  + 1)
            if c < max_neighb_tries:
                final_points[offset+n_added*2+1] = new_point
                n_added += 1
            # else: print(point, "failed")
            rot = np.delete(rot, new_dir, axis=0)
            if len(rot) == 0 and n_added < 3:
                # print("last call", point)
                # print("bad point", i, "; offset", offset, "; n_added", n_added)
                # len_idx = len(fail_idx)
                fail_idx += [offset + (x + n_added) * 2     for x in range(num_of_neighbours - n_added)]
                fail_idx += [offset + (x + n_added) * 2 + 1 for x in range(num_of_neighbours - n_added)]
                # print(fail_idx[len_idx:])
                break
                # print("Failed to find Neighbour")
    # print("Fails:", len(fail_idx), f"({len(set(fail_idx))}) | expected {len(final_points) - len(fail_idx)}")
    final_points = np.delete(final_points, fail_idx, axis=0)

    return final_points

def generate_square(num_points_per_side, side_length, center_point, rotations=(0, 0, 0), start_point=()):
    # in xy plane
    points = []
    for i in range(num_points_per_side):
        t = i / (num_points_per_side - 1)
        points.append([-0.5 * side_length, -0.5 * side_length + t * side_length, 0])
    for i in range(1, num_points_per_side):
        t = i / (num_points_per_side - 1)
        points.append([-0.5 * side_length + t * side_length, 0.5 * side_length, 0])
    for i in range(1, num_points_per_side):
        t = i / (num_points_per_side - 1)
        points.append([0.5 * side_length, 0.5 * side_length - t * side_length, 0])
    for i in range(1, num_points_per_side):
        t = i / (num_points_per_side - 1)
        points.append([0.5 * side_length - t * side_length, -0.5 * side_length, 0])
    points = np.array(points)
    
    # Rotate the points around the origin (0, 0, 0)
    r = R.from_euler('xyz', rotations, degrees=True)
    points_rotated = r.apply(points)
    
    # Translate the rotated points to the given center point
    # center_x, center_y, center_z = center_point
    points_translated = points_rotated + np.array(center_point)

    if start_point:
        if len(start_point) == 3: start_point = (*start_point, 5)
        points_translated = np.insert(points_translated, 0, [points_translated[0] - (points_translated[0] - start_point[:3]) / start_point[3] * (start_point[3] - i) for i in range(start_point[3])], axis=0)
    
    return points_translated

def generate_circle(num_points, radius, center_point, rotations, start_point=()):
    points = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Rotate the points around the origin (0, 0, 0)
    r = R.from_euler('xyz', rotations, degrees=True)
    points_rotated = r.apply(points)
    
    # Translate the rotated points to the given center point
    points_translated = points_rotated + np.array(center_point)

    if start_point:
        if len(start_point) == 3: start_point = (*start_point, 5)
        points_translated = np.insert(points_translated, 0, [points_translated[0] - (points_translated[0] - start_point[:3]) / start_point[3] * (start_point[3] - i) for i in range(start_point[3])], axis=0)    
    return points_translated

def generate_spiral(num_points, max_radius, center_point, num_revolutions=4, start_from_center=True, start_point=(), reverse=False, rotations=(0, 0, 0)):
    points = []
    
    if reverse:
        theta_increment = -2 * np.pi * num_revolutions / num_points
    else:
        theta_increment = 2 * np.pi * num_revolutions / num_points

    for i in range(num_points):
        theta = i * theta_increment
        radius = max_radius * (i / num_points)  # Linearly increase radius
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0
        points.append([x, y, z])

    points = np.array(points)
    
    # Rotate the points around the origin (0, 0, 0)
    r = R.from_euler('xyz', rotations, degrees=True)
    points_rotated = r.apply(points)
    
    # Translate the rotated points to the given center point
    points_translated = points_rotated + np.array(center_point)
    
    if start_from_center:
        if len(start_point) == 3: start_point = (*start_point, 5)
        points_translated = np.insert(points_translated, 0, [points_translated[0] - (points_translated[0] - start_point[:3]) / start_point[3] * (start_point[3] - i) for i in range(start_point[3])], axis=0)
        return points_translated
    else:
        points_translated = np.flip(points_translated, axis=0)
        if len(start_point) == 3: start_point = (*start_point, 5)
        return np.insert(points_translated, 0, [points_translated[0] - (points_translated[0] - start_point[:3]) / start_point[3] * (start_point[3] - i) for i in range(start_point[3])], axis=0)

def generate_coil(num_points, radius, height, num_turns, starting_point=(0, 0, 0), rotations=(0, 0, 0), starts=(), spread_xy=1):
    theta = np.linspace(0, 2*np.pi*num_turns, num_points)
    z = np.linspace(0, height, num_points)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    coil_points = np.column_stack((x, y, z))
    
    rotation = R.from_euler('xyz', rotations, degrees=True)
    points_translated = rotation.apply(coil_points) + starting_point

    scaling = np.hstack((
        np.linspace(1, spread_xy, ceil(points_translated.shape[0] / 2)),
        np.linspace(spread_xy, 1, round(points_translated.shape[0] / 2)),
        ))
    points_translated[:, :2] *= scaling[:, np.newaxis]
    
    if starts:
        if len(starts) == 3: starts = (*starts, 5)
        points_translated = np.insert(points_translated, 0, [points_translated[0] - (points_translated[0] - starts[:3]) / starts[3] * (starts[3] - i) for i in range(starts[3])], axis=0)

    return points_translated
    
def plot_pts(points, quiv=False, labels=False):
    import matplotlib.pyplot as plt

    x, y, z = zip(*points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.plot(x, y, z, c='b', marker='.')

    if quiv:
        # Calculate vectors between points
        vectors = np.diff(points, axis=0)

        # Extract the components of vectors
        u, v, w = zip(*vectors)

        # Plot vectors as quivers
        ax.quiver(x[:-1], y[:-1], z[:-1], u, v, w, color='r')

    if labels:
        for i, p in enumerate(points):
            ax.text(*p, str(i+1))

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


if __name__ == "__main__":
    # print(simple_act_cloud()[:5])
    
    # random_points = generate_random_points(750, 2250, 150, 100) # in actuator space
    # print(random_points[:5])
    # curv_pts = curv_up_down_seq(750, 2250, 6, 2)
    # curv_pts = curv_test(750, 2250)
    # curv_pts = curv_test(800, 2200, move_num=7, servos=[0,1,2], fixed_pos=[800, 1220, 1800], reps=1)
    # print(curv_pts.shape)
    # print(curv_pts[:])
    # grid_pts = grid(750, 2250, 4, randomize=True)
    # print("grid_pts:", len(grid_pts))
    # print(grid_pts[:20])

    # new_grid_pts = add_neighbours(grid_pts[:len(grid_pts)//3], 3, 50, 1500, grid=True)
    # new_grid_pts = add_neighbours(random_points, 3, 150, 1000)
    # print("grid_pts w/ neighbours:", len(new_grid_pts))
    # print(new_grid_pts[:20])
    # random_points.insert(0, [65, 65, 65])
    # nums = [9,7,10,8]
    # traj = vert_up_down_seq(750,2250,nums)
    # print(traj)
    # print(traj[:nums[0]*2])
    # print([traj[sum(traj[i-1])*2-1:sum(traj[i])*2-1] for i in nums])
    # with open('./data/waypoints.txt', 'w') as file:
    #     for point in random_points:
    #         file.write(f'{point[0]}, {point[1]}, {point[2]}\n')
    
    # plot_pts(generate_spiral(60, 45, (0, 0, 110), rotations=(0, 0, 0), start_from_center=False), labels=True)
    # plot_pts(generate_coil(120, 20, 120, 5, starting_point=(0, 0, 110), rotations=(0, 90, 0)), labels=True)
    plot_pts(generate_circle(25, 35, (0, 20, 110), (0, 35, 45), start_point=(0, 0, 125.5, 3)), labels=True)
    plt.show()

    # print(len(generate_circle(40, 25, (30, 20, 110), (10, 80, 45))))
