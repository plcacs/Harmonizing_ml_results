import json
import os
import uuid
from typing import List, Dict, Optional, Any
from aws_cdk import aws_s3_assets as assets, cloudformation_include, aws_iam as iam, aws_lambda as lambda_
try:
    from aws_cdk.core import Construct
    from aws_cdk import core as cdk
except ImportError:
    import aws_cdk as cdk
    from constructs import Construct
from chalice import api

class Chalice(Construct):
    """Chalice construct for CDK.

    Packages the application into AWS SAM format and imports the resulting
    template into the construct tree under the provided ``scope``.

    """

    def __init__(self, scope: Union[str, bool, dict], id: Union[str, None], source_dir: Union[str, int], stage_config: Union[None, str, dict, dict[str, typing.Any]]=None, preserve_logical_ids: bool=True, **kwargs) -> None:
        """Initialize Chalice construct.

        :param str source_dir: Path to Chalice application source code.
        :param dict stage_config: Chalice stage configuration.
            The configuration object should have the same structure as Chalice
            JSON stage configuration.
        :param bool preserve_logical_ids: Whether the resources should have
            the same logical IDs in the resulting CDK template as they did in
            the original CloudFormation template file. If you're vending a
            Construct using cdk-chalice, make sure to pass this as ``False``.
            Note: regardless of whether this option is true or false, the
            :attr:`sam_template`'s ``get_resource`` and related methods always
            uses the original logical ID of the resource/element, as specified
            in the template file.
        :raises `ChaliceError`: Error packaging the Chalice application.
        """
        super(Chalice, self).__init__(scope, id, **kwargs)
        self.source_dir = os.path.abspath(source_dir)
        self.stage_name = scope.to_string()
        self.stage_config = stage_config
        chalice_out_dir = os.path.join(os.getcwd(), 'chalice.out')
        package_id = uuid.uuid4().hex
        self._sam_package_dir = os.path.join(chalice_out_dir, package_id)
        self._package_app()
        sam_template_filename = self._generate_sam_template_with_assets(chalice_out_dir, package_id)
        self.sam_template = cloudformation_include.CfnInclude(self, 'ChaliceApp', template_file=sam_template_filename, preserve_logical_ids=preserve_logical_ids)
        self._function_cache = {}
        self._role_cache = {}

    def _package_app(self) -> None:
        api.package_app(project_dir=self.source_dir, output_dir=self._sam_package_dir, stage=self.stage_name, chalice_config=self.stage_config)

    def _generate_sam_template_with_assets(self, chalice_out_dir: Union[str, bytes], package_id: Union[str, bytes]) -> str:
        deployment_zip_path = os.path.join(self._sam_package_dir, 'deployment.zip')
        sam_deployment_asset = assets.Asset(self, 'ChaliceAppCode', path=deployment_zip_path)
        sam_template_path = os.path.join(self._sam_package_dir, 'sam.json')
        sam_template_with_assets_path = os.path.join(chalice_out_dir, '%s.sam_with_assets.json' % package_id)
        with open(sam_template_path) as sam_template_file:
            sam_template = json.load(sam_template_file)
            for function in self._filter_resources(sam_template, 'AWS::Serverless::Function'):
                function['Properties']['CodeUri'] = {'Bucket': sam_deployment_asset.s3_bucket_name, 'Key': sam_deployment_asset.s3_object_key}
            managed_layers = self._filter_resources(sam_template, 'AWS::Serverless::LayerVersion')
            if len(managed_layers) == 1:
                layer_filename = os.path.join(self._sam_package_dir, 'layer-deployment.zip')
                layer_asset = assets.Asset(self, 'ChaliceManagedLayer', path=layer_filename)
                managed_layers[0]['Properties']['ContentUri'] = {'Bucket': layer_asset.s3_bucket_name, 'Key': layer_asset.s3_object_key}
        with open(sam_template_with_assets_path, 'w') as f:
            f.write(json.dumps(sam_template, indent=2))
        return sam_template_with_assets_path

    def _filter_resources(self, template: str, resource_type: Union[str, typing.Type]) -> list:
        return [resource for resource in template['Resources'].values() if resource['Type'] == resource_type]

    def get_resource(self, resource_name: str) -> str:
        return self.sam_template.get_resource(resource_name)

    def get_role(self, role_name: Union[str, None]):
        if role_name not in self._role_cache:
            cfn_role = self.sam_template.get_resource(role_name)
            role = iam.Role.from_role_arn(self, role_name, cfn_role.attr_arn)
            self._role_cache[role_name] = role
        return self._role_cache[role_name]

    def get_function(self, function_name: str):
        if function_name not in self._function_cache:
            cfn_lambda = self.sam_template.get_resource(function_name)
            arn_ref = cfn_lambda.get_att('Arn')
            function = lambda_.Function.from_function_arn(self, function_name, arn_ref.to_string())
            self._function_cache[function_name] = function
        return self._function_cache[function_name]

    def add_environment_variable(self, key: Union[str, typing.Sequence[str], typing.Callable[KT, VT]], value: Union[str, typing.Sequence[str], typing.Callable[KT, VT]], function_name: Union[str, None]) -> None:
        cfn_function = self.sam_template.get_resource(function_name)
        cfn_function.add_override('Properties.Environment.Variables.%s' % key, value)