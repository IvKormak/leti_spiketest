from os import mkdir
import itertools

from model import *

def save_output_to_file(indicator, journals):
    xdata, ydata = [], []
    xdata = range(len(journals[0]))
    for data in journals:
        ydata.append([k for k in data])
    ydata = [[ydata[j][i] for j in range(len(ydata))] for i in range(len(ydata[0]))]
    plt.plot(xdata, ydata)
    plt.savefig(str(indicator)+'.png')
    plt.close()

def sweeper_generator(data):
    def linspace(start, stop, n):
        if n == 1:
            yield stop
            return
        h = (stop - start) / (n - 1)
        for i in range(n):
            yield start + h * i

    inner = dict(**data)
    iters = (linspace(*value) for value in data.values())

    for iter in itertools.product(*iters):
        inner.update(zip(data.keys(), iter))
        yield inner

if __name__ == "__main__":

    number_of_categories = 7

    generation_num = 3
    specimen_num = 5 #>=2
    traces_num = 100

    param_static = {
        'i_thres': 5000,
        'w_min': 1,
        'w_max': 1000,
        'a_dec': 50,
        'a_inc': 150,
        't_leak': 800,
        't_ltp': 1200,
        'randmut': 0.001
    }
    param_iter = {
        't_refrac': (80 * 10 ** 2, 1 * 10 ** 3, 10),
        't_inhibit': (70 * 10 ** 1, 1 * 10 ** 2, 10)
    }

    param_set_sweep = sweeper_generator(param_iter)
    total_sweep_steps = len(list(sweeper_generator(param_iter)))
    sweep_step = 0

    param_set_best = {}
    first_set = next(param_set_sweep)
    first_set.update(param_static)

    model = init_model(learn = True, teacher = True, structure = [number_of_categories,])

    folder = f"exp{int(time.time())}"
    mkdir(folder)

    best_scores = []
    best_score = 0
    best_set = {}

    for set in param_set_sweep:
        set.update(param_static)
        sweep_step += 1
        starttime = time.time()
        print(f"step {sweep_step} of {total_sweep_steps}")
        best_weights = []

        for i in range(generation_num):
            # number of generations
            weights = []
            scores = []
            journals = []
            traces = []
            for j in range(specimen_num):

                timer.reset()

                #number of specimen in generation
                traces = [random.choice(Defaults.files) for _ in range(traces_num)]
                model.mutate(*best_weights)
                tot = 0
                for trace in traces:
                    #number of traces to train with
                    model.feed.load(trace)
                    while frame := model.next():
                        model.teacher.output = [1 if n == Defaults.files.index(trace) else 1j
                                                      for n in range(number_of_categories)]
                    """for frame in model:
                        model.teacher.output = [1 if n == Defaults.files.index(trace) else 1j
                                                for n in range(number_of_categories)]"""
                    tot += model.frame

                weights.append(model.get_weights())

                fitness = model.calculate_fitness()
                scores.append(fitness[0])
                #print(time.time() - starttime, fitness[0:])

            s_scores = list(reversed(sorted(scores)))
            best_scores.append(s_scores[0])
            if s_scores[0] > best_score:
                best_score = s_scores[0]
                best_set = set.copy()

            index1 = scores.index(s_scores[0])
            index2 = scores.index(s_scores[1])
            best_weights = (weights[index1], weights[index2])

        model.set_param_set(set)
        print(f"time spent {time.time() - starttime}")
        pickle.dump({'set': best_set, 'score': best_score}, open(f"{folder}/optimal.param", "wb"))

    print (f"best value: {best_set}\n fitness:{best_score}")

    plt.plot(list(range(len(best_scores))), best_scores)
    plt.savefig(f"{folder}/graph.png")
    pickle.dump(best_weights[0], open(f"{folder}/f.weight", "wb"))
    model.save_attention_maps(folder)