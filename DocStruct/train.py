import logging

from utils.average_utils import *
from utils.linklink_utils import *


def validate(model, loss_fn, eval_fn, val_loader, config):
    model.eval()

    world_size = link.get_world_size()

    with torch.no_grad():
        val_loss, val_eval = {}, {}
        for r in eval_fn.result:
            val_eval[r] = Average(len(val_loader))
        for r in loss_fn.loss:
            val_loss[r] = Average(len(val_loader))

        for batch in val_loader:
            batch = batch.to(device=config.run.device)
            output = model(batch)
            loss = loss_fn(output, batch)
            eval = eval_fn(output, batch)

            for r in loss_fn.loss:
                val_loss[r].update(loss[r].item())
            for r in eval_fn.result:
                val_eval[r].update(eval[r])

        for r in eval_fn.result:
            val_eval[r] = torch.tensor(val_eval[r].average / world_size)
            link.allreduce(val_eval[r])
            val_eval[r] = val_eval[r].item()
        for r in loss_fn.loss:
            val_loss[r] = torch.tensor(val_loss[r].average / world_size)
            link.allreduce(val_loss[r])
            val_loss[r] = val_loss[r].item()

    # Question: Why put model.train() here?
    model.train()

    return val_loss, val_eval


def train(model, loss_fn, eval_fn, train_loader, train_val_loader, val_loader, config, root_path):
    rank, world_size = link.get_rank(), link.get_world_size()

    best_eval = float('-inf')
    best_iter = -1

    optimizer = model.module.get_optimizer(True)

    logging.info("Training begin ...")
    msg = "Iter: [{}/%d] val [{}] [{}] best [{:.4f}] {}" % config.run.total_iter
    # msg = "Iter: [{}/%d] train [{}] [{}] val [{}] [{}] best [{:.4f}] {}" % config.run.total_iter

    try:
        model.train()

        eval_mode = 'joint'
        eval_fn.evaluators['Reconstruct'].switch_mode(eval_mode)
        eval_fn.evaluators['TopKAcc'].switch_mode(eval_mode)

        for i, batch in zip(range(1, len(train_loader) + 1), train_loader):

            batch = batch.to(device=config.run.device)
            output = model(batch)
            loss = loss_fn(output, batch)

            optimizer.zero_grad()
            loss['loss'].backward()
            reduce_gradients(model, config.linklink.sync)
            optimizer.step(epoch=i)

            if i % config.run.print_per_iter != 0:
                continue

            val_loss, val_eval = validate(model, loss_fn, eval_fn, val_loader, config)
            # train_loss, train_eval = validate(model, loss_fn, eval_fn, train_val_loader, config)

            if val_eval['eval'] > best_eval:
                best = '*'
                best_eval = val_eval['eval']
                best_iter = i
                if rank == 0 and config.run.mode != 'test':
                    torch.save(model.state_dict(), os.path.join(root_path, 'model.pt'))
            else:
                best = ''

            # Question: to HX, why logging doesn't need rank == 0
            #  Answer: if rank!=0, its logging doesn't have printing handler
            if config.tensorboard is not None and rank == 0:
                for r in loss_fn.loss:
                    # config.tensorboard.add_scalar(f'train/loss/{r}', train_loss[r], i)
                    config.tensorboard.add_scalar(f'valid/loss/{r}', val_loss[r], i)
                for r in eval_fn.result:
                    # config.tensorboard.add_scalar(f'train/eval/{r}', train_eval[r], i)
                    config.tensorboard.add_scalar(f'valid/eval/{r}', val_eval[r], i)

            # train_loss_msg = ' '.join('{}: {:.4f}'.format(k, v) for k, v in train_loss.items())
            val_loss_msg = ' '.join('{}: {:.4f}'.format(k, v) for k, v in val_loss.items())

            # train_eval_msg = ' '.join('{}: {:.4f}'.format(k, v) for k, v in train_eval.items())
            val_eval_msg = ' '.join('{}: {:.4f}'.format(k, v) for k, v in val_eval.items())

            logging_info = msg.format(i, val_loss_msg, val_eval_msg, best_eval, best)
            logging.info(logging_info)

    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt best val eval: {:.4f} at iter: [{}/{}]'.format(best_eval,
                                                                                       best_iter,
                                                                                       config.run.total_iter))

    logging.info('Train end best val eval: {:.4f} at iter: [{}/{}]'.format(best_eval, best_iter, config.run.total_iter))

    link.barrier()


