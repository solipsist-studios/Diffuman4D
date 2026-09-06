from queue import Queue, Empty
from threading import Thread
import traceback

from src.samplers.sliding_iterative_sampler import SlidingIterativeSampler
from src.samplers.utils.sampling_utils import check_sampling_results
from src.data.utils.metric_utils import evaluate_results
from src.utils import RankedLogger

from scripts.nerfstudio.diffuman4d_to_nerfstudio import diffuman4d_to_nerfstudio

log = RankedLogger(__name__, rank_zero_only=True)


class SamplingRunner:
    def __init__(self, sampler: SlidingIterativeSampler):
        self.sampler = sampler

    def prepare_task_queues(self):
        self.task_queues = []
        for tasks in self.sampler.all_tasks:
            task_queue = Queue()
            for task in tasks:
                task_queue.put(task)
            self.task_queues.append(task_queue)

    def parallel_execute_tasks(self, task_queue: Queue):
        exceptions = []  # Store exceptions from worker threads
        
        def _worker(task_queue: Queue, pipe_idx: int):
            while True:
                try:
                    task = task_queue.get_nowait()
                except Empty:
                    break
                
                try:
                    self.sampler.execute_one_task(task, pipe_idx=pipe_idx)
                except Exception as e:
                    error_msg = f"Error in worker {pipe_idx} processing task {task}: {str(e)}"
                    log.error(error_msg)
                    log.error(f"Full traceback:\n{traceback.format_exc()}")
                    exceptions.append((task, e, traceback.format_exc()))

        # create threads for each pipeline
        threads = [Thread(target=_worker, args=(task_queue, i)) for i in range(len(self.sampler.pipelines))]

        # start threads
        for thread in threads:
            thread.start()
        # wait for all threads to finish
        for thread in threads:
            thread.join()
        
        # Check if any exceptions occurred
        if exceptions:
            log.error(f"Encountered {len(exceptions)} error(s) during parallel execution:")
            for task, exc, tb in exceptions:
                log.error(f"Task: {task}")
                log.error(f"Exception: {exc}")
            raise RuntimeError(f"Parallel execution failed with {len(exceptions)} error(s). See logs above for details.")

    def inference(self):
        if check_sampling_results(
            self.sampler.spa_labels, self.sampler.tem_labels, output_dir=self.sampler.output_dir
        ):
            log.info("Sampling results already exist. Skipping inference.")
            return

        log.info(
            f"Starting to execute tasks on {len(self.sampler.pipelines)} GPUs. "
            f"The results will be saved in {self.sampler.output_dir}."
        )
        if len(self.sampler.pipelines) > 1:
            self.prepare_task_queues()

            for i, task_queue in enumerate(self.task_queues):
                log.info(f"Executing tasks (Altenation round: {i + 1}/{len(self.task_queues)}).")
                self.parallel_execute_tasks(task_queue)

            if not check_sampling_results(
                self.sampler.spa_labels, self.sampler.tem_labels, output_dir=self.sampler.output_dir
            ):
                raise ValueError("Sampling failed.")
        else:
            self.sampler.execute_tasks()

    def evaluate(self):
        evaluate_results(
            pred_images_dir=f"{self.sampler.output_dir}/images",
            gt_images_dir=f"{self.sampler.dataset.data_dir}/{self.sampler.dataset.scene_label}/images",
            fmasks_dir=f"{self.sampler.dataset.data_dir}/{self.sampler.dataset.scene_label}/fmasks",
            pred_image_ext=".jpg",
            gt_image_ext=".webp",
            fmask_ext=".png",
            spa_labels=self.sampler.target_spa_labels,
            tem_labels=self.sampler.tem_labels,
            out_metrics_path=f"{self.sampler.output_dir}/metrics.json",
            crop_with_fmask=True,
            background_color="white",
        )

    def to_nerfstudio(self):
        diffuman4d_to_nerfstudio(
            data_dir=f"{self.sampler.dataset.data_dir}/{self.sampler.dataset.scene_label}",
            result_dir=self.sampler.output_dir,
            input_cameras=self.sampler.input_spa_labels,
            tem_label=self.sampler.tem_labels[0] if self.sampler.tem_labels else None,
        )
