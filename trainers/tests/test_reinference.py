# trainers/tests/test_reinference.py
"""
Copyright (C) 2026 by the CAM-AI team, info@cam-ai.de
More information and complete source: https://github.com/ludgerh/cam-ai
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
"""

import asyncio
import os
import tempfile
from logging import getLogger
from threading import Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import cv2 as cv
import numpy as np
from asgiref.sync import sync_to_async
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TransactionTestCase
from django.utils import timezone

from tf_workers.c_tf_workers import model_buffer, tf_worker, tf_worker_client
from tf_workers.models import school, worker
from trainers import c_trainers
from trainers.c_trainers import trainer
from trainers.models import trainer as dbtrainer
from trainers.models import trainframe


class WorkerRequestCorrelationTests(SimpleTestCase):

  async def test_model_buffer_keeps_correlated_request_key(self):
    buffer = model_buffer(1)
    buffer.set_xy(1, 1)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    worker_process = SimpleNamespace(
      check_model=AsyncMock(),
      models={1: {'xdim': 1}},
      logger=getLogger('worker-test'),
      logname='worker-test',
      users_lock=asyncio.Lock(),
      users={7: SimpleNamespace(prio=8)},
      model_buffers={1: buffer},
    )

    await tf_worker.process_received(
      worker_process,
      ['ask_pred', 7, 1, 'request-a', frame],
    )

    result = await buffer.get(1)
    self.assertEqual(result[0][1], (7, 'request-a'))
    np.testing.assert_array_equal(result[0][0], frame)

  async def test_correlated_result_is_published_to_registered_user_queue(self):
    buffer = model_buffer(1)
    buffer.pause = False
    frame = np.ones((1, 1, 3), dtype=np.uint8)
    await buffer.queues[8].put((frame, (7, 'request-a')))
    worker_process = SimpleNamespace(
      model_buffers = {1: buffer},
      dbline = SimpleNamespace(maxblock = 8, gpu_sim = 0.0),
      cachedict = {},
      len_taglist = 10,
      do_tf_debug = False,
      my_output = AsyncMock(),
      id = 1,
      _safe_maxblock = lambda: 8,
    )

    await tf_worker.process_buffer(worker_process, 1, getLogger('worker-test'))

    output = worker_process.my_output.put.await_args
    self.assertEqual(output.args[0], 7)
    self.assertEqual(output.args[1][2], (7, 'request-a'))

  async def test_late_response_for_finished_request_is_discarded(self):
    client = tf_worker_client(None, None)
    request_key = (7, 'request-a')
    client.my_output = SimpleNamespace(get=AsyncMock(side_effect=[
      ('pred_to_send', np.ones((1, 10)), request_key),
      ('stop',),
    ]))

    await client.out_reader_proc(7)

    self.assertNotIn(request_key, client.pred_out_dict)


class ReinferenceRecoveryTests(TransactionTestCase):

  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory(prefix = 'camai-reinference-')
    self.addCleanup(self.tempdir.cleanup)
    self.frames_dir = os.path.join(self.tempdir.name, 'frames')
    os.makedirs(self.frames_dir)

    self.user = User.objects.create(username = 'reinference-test')
    self.worker = worker.objects.create()
    self.school = school.objects.create(
      name = 'Re-inference test',
      creator = self.user,
      remote_creator = self.user,
      tf_worker = self.worker,
      dir = self.tempdir.name + os.sep,
      last_fit = 8,
      last_fit_infer = 7,
    )
    self.dbline = dbtrainer.objects.create(
      name = 'Re-inference test trainer',
      inference_brake = 0,
      inference_waitingtime = 0,
    )
    self.trainer = trainer(self.dbline.id, None, None, Lock())
    self.trainer.dbline = self.dbline
    self.trainer.logger = Mock()
    self.trainer.tf_w_index = 7
    self.trainer.tf_worker = AsyncMock()
    self.trainer.tf_worker.get_from_outqueue.return_value = np.full(
      (1, 10), 0.25, dtype = np.float32,
    )

    self.redis = Mock()
    self.redis.get_predict_proc_active.return_value = False
    redis_patch = patch.object(c_trainers, 'trainers_redis', self.redis)
    redis_patch.start()
    self.addCleanup(redis_patch.stop)

  def make_frames(self, names):
    frames = []
    for name in names:
      values = {
        'made': timezone.now(),
        'school': self.school.id,
        'name': name,
        'code': 'NE',
        'checked': 1,
        'train_status': 0,
      }
      values.update({f'c{i}': int(i == 0) for i in range(10)})
      frames.append(trainframe(**values))
    return(trainframe.objects.bulk_create(frames))

  def write_frame(self, name, content):
    with open(os.path.join(self.frames_dir, name), 'wb') as frame_file:
      frame_file.write(content)

  async def test_corrupt_frame_is_skipped_and_next_frame_is_inferred(self):
    corrupt, valid = await sync_to_async(self.make_frames)([
      'corrupt.bmp',
      'valid.bmp',
    ])
    self.write_frame(corrupt.name, b'not an image')
    self.write_frame(valid.name, b'valid image bytes')
    image = np.zeros((2, 2, 3), dtype = np.uint8)

    with patch.object(
        c_trainers, 'decode_and_convert_image',
        side_effect = [cv.error('corrupt image'), image]):
      await self.trainer.update_predictions(self.school.id, self.school.last_fit)

    corrupt = await trainframe.objects.aget(id = corrupt.id)
    valid = await trainframe.objects.aget(id = valid.id)
    self.assertEqual(corrupt.last_fit, self.school.last_fit_infer)
    self.assertEqual(valid.last_fit, self.school.last_fit_infer)
    self.assertEqual(valid.pred0, 0.25)
    self.assertEqual(self.trainer.doing_reinference, 0)
    self.redis.set_predict_proc_active.assert_called_with(self.school.id, False)
    request_id = self.trainer.tf_worker.ask_pred.call_args.kwargs['request_id']
    self.assertEqual(
      self.trainer.tf_worker.get_from_outqueue.call_args.kwargs['request_id'],
      request_id,
    )
    self.trainer.tf_worker.finish_request.assert_awaited_with(7, request_id)

  async def test_missing_first_batch_advances_to_remaining_frames(self):
    frames = await sync_to_async(self.make_frames)([
      f'missing-{i}.bmp' for i in range(1001)
    ])

    await self.trainer.update_predictions(self.school.id, self.school.last_fit)

    self.assertEqual(
      await trainframe.objects.filter(
        school = self.school.id,
        last_fit = self.school.last_fit_infer,
      ).acount(),
      1000,
    )
    remaining = await trainframe.objects.aget(id = frames[-1].id)
    self.assertEqual(remaining.last_fit, -1)

    self.write_frame(remaining.name, b'valid image bytes')
    image = np.zeros((2, 2, 3), dtype = np.uint8)
    with patch.object(c_trainers, 'decode_and_convert_image', return_value = image):
      await self.trainer.update_predictions(self.school.id, self.school.last_fit)

    remaining = await trainframe.objects.aget(id = remaining.id)
    self.assertEqual(remaining.last_fit, self.school.last_fit_infer)
    self.assertEqual(self.trainer.tf_worker.ask_pred.await_count, 1)

  async def test_dropped_response_is_abandoned_and_next_scan_can_retry(self):
    frame = (await sync_to_async(self.make_frames)(['retry.bmp']))[0]
    self.write_frame(frame.name, b'valid image bytes')
    image = np.zeros((2, 2, 3), dtype = np.uint8)
    self.trainer.tf_worker.get_from_outqueue.side_effect = [None, None]

    with (
        patch.object(c_trainers, 'decode_and_convert_image', return_value = image),
        patch.object(c_trainers, 'INFERENCE_RESULT_ATTEMPTS', 2),
        patch.object(c_trainers, 'INFERENCE_RESULT_TIMEOUT', 0.01),
    ):
      await self.trainer.update_predictions(self.school.id, self.school.last_fit)

    frame = await trainframe.objects.aget(id = frame.id)
    self.assertEqual(frame.last_fit, -1)
    self.assertEqual(self.trainer.doing_reinference, 0)
    self.redis.set_predict_proc_active.assert_called_with(self.school.id, False)
    self.assertEqual(self.trainer.tf_worker.ask_pred.await_count, 1)
    failed_request = self.trainer.tf_worker.ask_pred.call_args.kwargs['request_id']
    self.trainer.tf_worker.finish_request.assert_awaited_with(7, failed_request)

    self.trainer.tf_worker.get_from_outqueue.side_effect = None
    self.trainer.tf_worker.get_from_outqueue.return_value = np.full(
      (1, 10), 0.5, dtype = np.float32,
    )
    with patch.object(c_trainers, 'decode_and_convert_image', return_value = image):
      await self.trainer.update_predictions(self.school.id, self.school.last_fit)

    frame = await trainframe.objects.aget(id = frame.id)
    self.assertEqual(frame.last_fit, self.school.last_fit_infer)
    self.assertEqual(self.trainer.tf_worker.ask_pred.await_count, 2)
    retried_request = self.trainer.tf_worker.ask_pred.call_args.kwargs['request_id']
    self.assertNotEqual(failed_request, retried_request)

  async def test_unexpected_worker_error_releases_locks_for_next_scan(self):
    frame = (await sync_to_async(self.make_frames)(['worker-error.bmp']))[0]
    self.write_frame(frame.name, b'valid image bytes')
    image = np.zeros((2, 2, 3), dtype = np.uint8)
    self.trainer.tf_worker.ask_pred.side_effect = RuntimeError('worker queue failed')

    with patch.object(c_trainers, 'decode_and_convert_image', return_value = image):
      with self.assertRaisesRegex(RuntimeError, 'worker queue failed'):
        await self.trainer.update_predictions(self.school.id, self.school.last_fit)

    self.assertEqual(self.trainer.doing_reinference, 0)
    self.redis.set_predict_proc_active.assert_called_with(self.school.id, False)
    self.redis.set_predict_proc_started.assert_called_with(self.school.id, False)
    failed_request = self.trainer.tf_worker.ask_pred.call_args.kwargs['request_id']
    self.trainer.tf_worker.finish_request.assert_awaited_with(7, failed_request)

    self.trainer.tf_worker.ask_pred.side_effect = None
    with patch.object(c_trainers, 'decode_and_convert_image', return_value = image):
      await self.trainer.update_predictions(self.school.id, self.school.last_fit)

    frame = await trainframe.objects.aget(id = frame.id)
    self.assertEqual(frame.last_fit, self.school.last_fit_infer)

  async def test_cancellation_releases_request_and_locks(self):
    frame = (await sync_to_async(self.make_frames)(['cancelled.bmp']))[0]
    self.write_frame(frame.name, b'valid image bytes')
    image = np.zeros((2, 2, 3), dtype = np.uint8)
    waiting_for_result = asyncio.Event()

    async def wait_for_result(*args, **kwargs):
      waiting_for_result.set()
      await asyncio.Event().wait()

    self.trainer.tf_worker.get_from_outqueue.side_effect = wait_for_result
    with patch.object(c_trainers, 'decode_and_convert_image', return_value = image):
      task = asyncio.create_task(
        self.trainer.update_predictions(self.school.id, self.school.last_fit),
      )
      await asyncio.wait_for(waiting_for_result.wait(), timeout = 1.0)
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    self.assertEqual(self.trainer.doing_reinference, 0)
    self.redis.set_predict_proc_active.assert_called_with(self.school.id, False)
    self.redis.set_predict_proc_started.assert_called_with(self.school.id, False)
    request_id = self.trainer.tf_worker.ask_pred.call_args.kwargs['request_id']
    self.trainer.tf_worker.finish_request.assert_awaited_with(7, request_id)
