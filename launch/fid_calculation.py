import sys
import torch

sys.path.append(".")
sys.path.append("..")

from argparse import ArgumentParser
from pathlib import Path
from metrics.metrics import metrics_registry
from datasets.transforms import transforms_registry
from datasets.datasets import AttrDataset
from utils.common_utils import tensor2im, setup_seed


setup_seed(777)


def inferece_fid_editing(opts):
    fid_metric = metrics_registry["fid"]()
    transform = transforms_registry[opts.transforms]().get_transforms()["transform_inference"]

    with_attribute = True
    attr_name = opts.attr_name

    if attr_name.endswith("_r"):
        attr_name = attr_name[:-2]
        with_attribute = False

    attr_dataset = AttrDataset(opts.orig_path, 
                               attr_name,
                               transform, 
                               opts.celeba_attr_table_pth, 
                               with_attribute)

    not_attr_dataset = AttrDataset(opts.synt_path, 
                                   attr_name,
                                   transform, 
                                   opts.celeba_attr_table_pth, 
                                   not with_attribute)

    print(f"Percent of Images of attribute {opts.attr_name} is "
                  f"{len(attr_dataset) / (len(attr_dataset) + len(not_attr_dataset))}")

    attr_images = []
    for attr_image in attr_dataset:
        img = tensor2im(attr_image).convert("RGB")
        attr_images.append(img)

    edited_images = []
    for not_attr_image in not_attr_dataset:
        img = tensor2im(not_attr_image).convert("RGB")
        edited_images.append(img)

    from_data_arg = {
        "inp_data": attr_images,
        "fake_data": edited_images,
        "paths": [],
    }

    _, fid_value, _ = fid_metric("", "", out_path="", from_data=from_data_arg)
    print(f"FID for {opts.attr_name} is {fid_value:.4f}")






    # for celeba_attr_name, (our_attr_name, edit_powers) in self.config.inference.fid_eiditing_map.items():
    #     for edit_power in edit_powers:
    #         print(f"Start calculating FID for {our_attr_name}, {edit_power} ")

    #         transform = transforms_registry[self.config.data.transform]().get_transforms()[
    #             "transform_inference"
    #         ]
    #         dataset_type = AttrDatasetMasked if self.config.train.masked else AttrDataset

    #         if celeba_attr_name not in  ["Young", "Smiling"]:
    #             attr_dataset = dataset_type(self.config.data.inference_dir, celeba_attr_name,
    #                                         transform, "../datasets/CelebAMask-HQ-attribute-anno.txt", True)

    #             not_attr_dataset = dataset_type(self.config.data.inference_dir, celeba_attr_name,
    #                                         transform, "../datasets/CelebAMask-HQ-attribute-anno.txt", False)
    #             # not_attr_dataset = dataset_type(f"../HFGI/celeba_edits_pap_pap_smiling_{edit_power}", celeba_attr_name,
    #             #                            transform, "../datasets/CelebAMask-HQ-attribute-anno.txt", False)
    #         else:                      
    #             # ------ reverse
    #             attr_dataset = dataset_type(self.config.data.inference_dir, celeba_attr_name,
    #                                         transform, "../datasets/CelebAMask-HQ-attribute-anno.txt", False)
    #             not_attr_dataset = dataset_type(self.config.data.inference_dir, celeba_attr_name,
    #                                        transform, "../datasets/CelebAMask-HQ-attribute-anno.txt", True)                        
    #             # not_attr_dataset = dataset_type(f"../HFGI/celeba_edits_pap_pap_smiling_{edit_power}", celeba_attr_name,
    #             #                          transform, "../datasets/CelebAMask-HQ-attribute-anno.txt", True)
    #             # ------ reverse
    #         print(f"our")
    #         print(f"Percent of Images of attribute {celeba_attr_name} is "
    #               f"{len(attr_dataset) / (len(attr_dataset) + len(not_attr_dataset))}")

    #         not_attr_dataloader = DataLoader(
    #             not_attr_dataset,
    #             batch_size=self.config.model.batch_size,
    #             shuffle=False,
    #             num_workers=self.config.model.workers,
    #         )

    #         self.result_pics = []
    #         self.log_pics = []
    #         self.log_edit_pics = []
    #         self.inverter_results = []
    #         self.input_pics = []
    #         self.masks = []
    #         self.paths = not_attr_dataset.paths
    #         self.edited_images = []
    #         self.attr_images = []
    #         self.inverter.eval()

    #         im_num = 0
    #         for input_batch in tqdm(not_attr_dataloader):
    #             #break
    #             with torch.no_grad():
    #                 if self.config.train.masked:
    #                     input_batch, mask = input_batch
    #                     mask = mask.to(self.device).float()
    #                     input_batch = input_batch.to(self.device).float()
    #                     input_cuda = input_batch, mask
    #                     for mask_ in mask:
    #                         self.masks.append(mask_)
    #                 else:
    #                     input_cuda = input_batch.to(self.device).float()
    #                 images, result_batch = self._run_on_batch(input_cuda)
    #                 edited_imgs = self._run_editing_on_batch(
    #                     result_batch, our_attr_name, [edit_power]
    #                 )

    #                 edited_imgs = edited_imgs.reshape(*images.shape)
    #                 for edited_im in edited_imgs:
    #                     img = tensor2im(edited_im.cpu())

    #                     memory_tmp = BytesIO()
    #                     img.save(memory_tmp, format="jpeg")
    #                     img = Image.open(memory_tmp).convert("RGB")
    #                     memory_tmp.close()

    #                     # im_name = not_attr_dataset.paths[im_num]
    #                     # im_name = im_name.split("/")[-1]
    #                     # img.save(f"celeba_edits/{im_name}")

    #                     self.edited_images.append(img)
    #                     im_num += 1
                        

    #         for attr_image in attr_dataset:
    #             if self.config.train.masked:
    #                 attr_image = attr_image[0]
    #                 #print(attr_image.size())
    #                 #assert 1 == 2
    #             img = tensor2im(attr_image)
                
    #             self.attr_images.append(Image.fromarray(np.array(img)).convert("RGB"))
                
    #         # for attr_image in not_attr_dataset:
    #         #     if self.config.train.masked:
    #         #         attr_image = attr_image[0]
    #         #     img = tensor2im(attr_image)
                
    #         #     self.edited_images.append(Image.fromarray(np.array(img)).convert("RGB"))

    #         from_data_arg = {
    #             "fake_data": self.attr_images,
    #             "inp_data": self.edited_images,
    #             "paths": [],
    #         }
    #         _, value, _ = metric(
    #             self.config.data.inference_dir,
    #             self.inference_results_dir,
    #             out_path="",
    #             from_data=from_data_arg,
    #         )
    #         print(f"FID for {celeba_attr_name} with power {edit_power} is {value:.4f}")
    #     continue
            
    #     self.edited_images = []
    #     for attr_image in not_attr_dataset:
    #         if self.config.train.masked:
    #             attr_image = attr_image[0]
    #             #print(attr_image.size())
    #             #assert 1 == 2
    #         img = tensor2im(attr_image)
            
    #         self.edited_images.append(Image.fromarray(np.array(img)).convert("RGB"))
    #     from_data_arg = {
    #         "fake_data": self.attr_images,
    #         "inp_data": self.edited_images,
    #         "paths": [],
    #     }
    #     _, value2, _ = metric(
    #         self.config.data.inference_dir,
    #         self.inference_results_dir,
    #         out_path="",
    #         from_data=from_data_arg,
    #     )
    #     print(f"FID for {celeba_attr_name} no editing is {value2:.4f}")
        
        
    #     self.edited_images = self.attr_images[:len(self.attr_images)//2]
    #     self.attr_images = self.attr_images[len(self.attr_images)//2:]
    #     from_data_arg = {
    #         "fake_data": self.attr_images,
    #         "inp_data": self.edited_images,
    #         "paths": [],
    #     }
    #     _, value3, _ = metric(
    #         self.config.data.inference_dir,
    #         self.inference_results_dir,
    #         out_path="",
    #         from_data=from_data_arg,
    #     )
    #     print(f"FID for {celeba_attr_name} attr_images / 2 is {value3:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--orig_path", type=str, help="Path to directory of original Celeba images "
    )
    parser.add_argument(
        "--synt_path",
        type=str,
        help="Path to synthesized edited images",
    )
    parser.add_argument(
        "--attr_name",
        type=str,
        help="Name of Celeba attribute that is added during editing.'_r' at the end of attribute name means that attribute was not added but removed during editing",
    )
    parser.add_argument(
        "--celeba_attr_table_pth",
        type=str,
        help="Path to celeba attributes .txt",
    )
    parser.add_argument(
        "--transforms",
        default="hyperinv",
        type=str,
        help="Which transforms from datasets.transforms.transforms_registry should be used",
    )

    opts = parser.parse_args()
    inferece_fid_editing(opts)
